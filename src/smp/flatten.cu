#include "utils/utils.hpp"
#include <smp/flatten.cuh>
#include <thread>

void FlattenRowHost(const PreferenceLists &pls, int *flat_pl, int row, int n) {
  for (int i = 0; i < n; i++) {
    int a = pls[row][i];
    flat_pl[row * n + i] = a;
  }
}

int *ParallelFlattenHost(const PreferenceLists &pl, int n) {
  int *flat_pl;
  CUDA_CHECK(
      cudaMallocHost(&flat_pl, n * n * sizeof(int), cudaHostAllocDefault));

  int num_threads = std::thread::hardware_concurrency(); // Get the number of
                                                         // supported threads
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (int i = 0; i < n; ++i) {
    if (threads.size() >= num_threads) {
      // Wait for all the threads in this batch to complete.
      for (auto &th : threads) {
        th.join();
      }
      threads.clear();
    }

    threads.emplace_back(FlattenRowHost, std::ref(pl), flat_pl, i, n);
  }

  // Join the threads
  for (auto &th : threads) {
    th.join();
  }

  return flat_pl;
}