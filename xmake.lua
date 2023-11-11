set_languages("c++20")
add_requires("libtorch", "libcurl")

target("mnist")
  set_kind("binary")
  add_files("image-recognition/MNIST/*.cpp")
  add_packages("libtorch", "libcurl")
