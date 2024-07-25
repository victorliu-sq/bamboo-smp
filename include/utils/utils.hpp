#pragma once

#include <chrono>
#include <iostream>
#include <vector>

#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__                                 \
                << " with err msg: " << cudaGetErrorString(rt) << std::endl;   \
      throw;                                                                   \
    }                                                                          \
  } while (0);

static uint64_t getNanoSecond() {
  return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

using PreferenceLists = std::vector<std::vector<int>>;
