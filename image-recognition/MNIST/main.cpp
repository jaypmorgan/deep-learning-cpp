#include <filesystem>
#include <cstdlib>
#include <iostream>
#include <torch/torch.h>
#include <curl/curl.h>

struct LeNet : torch::nn::Module {

  torch::nn::Linear dense1{nullptr}, dense2{nullptr}, dense3{nullptr};
  torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
  torch::nn::AvgPool2d pool1{nullptr}, pool2{nullptr};

  LeNet() {
    conv1 = register_module(
      "conv1", 
      torch::nn::Conv2d(
        torch::nn::Conv2dOptions(1, 6, 5)
        .stride(1)
        .padding(2)
        .bias(false)));
    pool1 = register_module(
      "pool1",
      torch::nn::AvgPool2d(
        torch::nn::AvgPool2dOptions(2).stride(2)));
    conv2 = register_module(
      "conv2", 
      torch::nn::Conv2d(
        torch::nn::Conv2dOptions(6, 16, 5)
        .stride(1)
        .bias(false)));
    pool2 = register_module(
      "pool2",
      torch::nn::AvgPool2d(
        torch::nn::AvgPool2dOptions(2).stride(2)));
    dense1 = register_module(
      "dense1",
      torch::nn::Linear(400, 120));
    dense2 = register_module(
      "dense2",
      torch::nn::Linear(120, 84));
    dense3 = register_module(
      "dense3",
      torch::nn::Linear(84, 10));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = pool1->forward(torch::relu(conv1->forward(x)));  // sigmoid was unstable
    x = pool2->forward(torch::relu(conv2->forward(x)));
    x = x.flatten(1);
    x = torch::relu(dense1->forward(x));
    x = torch::relu(dense2->forward(x));
    x = torch::log_softmax(dense3->forward(x), 1);
    return x;
  }
};

static size_t write_data(void *ptr, size_t size, size_t nmemb, void *stream)
{
  size_t written = fwrite(ptr, size, nmemb, (FILE *)stream);
  return written;
}

int extract_file(std::string filename)
{
  std::string command {"gzip -d " + filename};
  int return_code = system(command.c_str());  // cheating??? Probably...
  return return_code;
}

void download_mnist(std::string root)
{
  // create the root directory
  std::filesystem::create_directory(root);

  // define the remote URL and resources
  CURL *curl_handle;
  std::string base_url = "http://yann.lecun.com/exdb/mnist/";
  std::vector<std::string> resources = {
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"};
  FILE *page_file;

  if (std::filesystem::exists(root+"/"+resources.back())
      || std::filesystem::exists(root+"/"+resources.back().substr(0, resources.back().length()-3))) {
    return;
  }

  std::cout << "Download MNIST data" << '\n';

  // iterate over the resources and download them
  curl_global_init(CURL_GLOBAL_ALL);
  for (auto& resource : resources) {
    curl_handle = curl_easy_init();
    curl_easy_setopt(curl_handle, CURLOPT_URL, (base_url+resource).c_str());
    curl_easy_setopt(curl_handle, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl_handle, CURLOPT_NOPROGRESS, 0L);
    page_file = fopen((root+"/"+resource).c_str(), "wb");
    if (page_file) {
      curl_easy_setopt(curl_handle, CURLOPT_WRITEDATA, page_file);
      curl_easy_perform(curl_handle);
      fclose(page_file);
    }
    curl_easy_cleanup(curl_handle);
    // extract the .gz file
    extract_file(root+"/"+resource);
  }
  curl_global_cleanup();
}

torch::Tensor accuracy(torch::Tensor prediction, torch::Tensor target)
{
  auto n_correct = torch::sum(prediction == target);
  return n_correct / target.size(0);
}

int main()
{
  // Download and create the dataset
  download_mnist("./data");
  auto data_loader = torch::data::make_data_loader(
    torch::data::datasets::MNIST("./data").map(
      torch::data::transforms::Stack<>()),
    64);

  // Create the executation device
  auto device {torch::kCPU};
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;

  // Create model and setup optimiser
  auto model = std::make_shared<LeNet>();
  model->to(device);
  torch::optim::SGD optimiser(model->parameters(), /*lr=*/0.01);

  // Train LeNet
  for (size_t epoch = 1; epoch <= 20; epoch++) {
    size_t batch_idx {0};
    double average_loss {0.0};
    for (auto& batch : *data_loader) {
      optimiser.zero_grad();
      torch::Tensor prediction = model->forward(batch.data.to(device));
      torch::Tensor loss = torch::nll_loss(prediction, batch.target.to(device));
      loss.backward();
      optimiser.step();
      average_loss += loss.item<double>();
      if (++batch_idx % 100 == 0) {
        std::cout << "Epoch: " << epoch << " "
                  << "Batch Idx: " << batch_idx << " "
                  << "Loss: " << average_loss/batch_idx << " "
                  << '\n';
      }
    }
  }

  // Save the trained model
  torch::save(model, "net.pt");
  
  // Load the test subset
  data_loader = torch::data::make_data_loader(
    torch::data::datasets::MNIST(
      "./data", torch::data::datasets::MNIST::Mode::kTest).map(
      torch::data::transforms::Stack<>()),
    64);

  // Create predictions and error rate
  model->eval();
  torch::Tensor acc {torch::tensor(0.0, device)};
  int steps {0};
  for (auto& batch : *data_loader) {
    torch::Tensor prediction = model->forward(batch.data.to(device));
    acc += accuracy(torch::argmax(prediction, 1), batch.target.to(device));
    ++steps;
  }
  std::cout << "Test Error: " << (1-(acc.item<float>()/steps))*100 << "%\n";

  return 0;
}
