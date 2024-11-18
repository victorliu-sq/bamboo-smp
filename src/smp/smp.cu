#include "smp/PRNode.cuh"
#include "utils/utils.hpp"
#include <fstream>
#include <iostream>
#include <smp/smp.cuh>
#include "../include/smp/flatten.cuh"
#include <sstream>
#include <vector>

namespace bamboosmp {
  SMP::SMP(const PreferenceLists &plM, const PreferenceLists &plW, const int &n) {
    flatten_pref_lists_m_ = ParallelFlattenHost(plM, n);
    flatten_pref_lists_w_ = ParallelFlattenHost(plW, n);
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
    std::vector<std::vector<int> > pref_lists_m;
    std::vector<std::vector<int> > pref_lists_w;

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

    auto smp = new SMP(pref_lists_m, pref_lists_w, n);

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
