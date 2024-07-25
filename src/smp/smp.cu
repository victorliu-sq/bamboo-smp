#include "smp/PRNode.cuh"
#include "utils/utils.hpp"
#include <fstream>
#include <iostream>
#include <smp/smp.cuh>
#include <sstream>
#include <vector>

namespace bamboosmp {

SMP::SMP(int *flatten_pref_lists_m, int *flatten_pref_lists_w)
    : flatten_pref_lists_m_(flatten_pref_lists_m),
      flatten_pref_lists_w_(flatten_pref_lists_w) {
  return;
}

SMP::~SMP() {
  CUDA_CHECK(cudaFreeHost(flatten_pref_lists_m_));
  CUDA_CHECK(cudaFreeHost(flatten_pref_lists_w_));
}

auto readPrefListsFromJson(const std::string &filepath) -> SMP * {
  std::ifstream file(filepath);
  if (!file) {
    std::cerr << "Could not open the file.\n";
    return nullptr;
  }

  std::string line;
  std::vector<std::vector<int>> pref_lists_m;
  std::vector<std::vector<int>> pref_lists_w;

  // Read men preferences
  std::getline(file, line); // Read "men:"
  while (std::getline(file, line)) {
    // std::cout << line << std::endl;
    if (line.empty())
      break;
    std::istringstream ss(line);
    int n;
    std::vector<int> temp;
    while (ss >> n) {
      // std::cout << n << std::endl;
      temp.push_back(n);
    }
    pref_lists_m.push_back(temp);
  }

  // Read women preferences
  std::getline(file, line); // Read "women:"
  while (std::getline(file, line)) {
    // std::cout << line << std::endl;
    if (line.empty())
      continue;
    std::istringstream ss(line);
    int n;
    std::vector<int> temp;
    while (ss >> n) {
      // std::cout << n << std::endl;
      temp.push_back(n);
    }
    pref_lists_w.push_back(temp);
  }

  int n = pref_lists_w.size();
  int *flatten_pref_lists_m;
  int *flatten_pref_lists_w;

  CUDA_CHECK(cudaMallocHost(&flatten_pref_lists_m, n * n * sizeof(int),
                            cudaHostAllocDefault));
  CUDA_CHECK(cudaMallocHost(&flatten_pref_lists_w, n * n * sizeof(int),
                            cudaHostAllocDefault));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      flatten_pref_lists_m[i * n + j] = pref_lists_m[i][j];
      flatten_pref_lists_w[i * n + j] = pref_lists_w[i][j];
    }
  }

  auto smp = new SMP(flatten_pref_lists_m, flatten_pref_lists_w);

  return smp;
}

__global__ void KernelAtomicMinTest(int n, int *word) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  printf("Hello from t %d\n", tid);
  if (tid < n) {
    int prev = atomicMin(word, tid);
    printf("[tid: %d]pred word's value is %d\n", tid, prev);
  }
}

__global__ void InitRankMatrix(int n, int num_blocks_per_list,
                               int num_threads_per_block, int *rank_mtx,
                               int *pref_list) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // row is man's idx and col is woman's idx
  int m_idx, w_idx, m_rank, w_rank;
  if (tid < n * n) {
    w_idx = tid / n;
    m_rank = tid % n;
    m_idx = pref_list[w_idx * n + m_rank];
    rank_mtx[w_idx * n + m_idx] = m_rank;
  }
}

__global__ void InitRankMatrixCol(int n, int num_blocks_per_list,
                                  int num_threads_per_block, int *rank_mtx,
                                  int *pref_list) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // row is man's idx and col is woman's idx
  int m_idx, w_idx, m_rank, w_rank;
  if (tid < n * n) {
    w_idx = tid / n;
    m_rank = tid % n;
    m_idx = pref_list[w_idx * n + m_rank];
    // rank_mtx[w_idx * n + m_idx] = m_rank;
    rank_mtx[m_idx * n + w_idx] = m_rank;
  }
}

} // namespace bamboosmp