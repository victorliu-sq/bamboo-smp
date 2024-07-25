#include <iostream>
#include <smp/BambooSMP.cuh>
#include <utils/utils.hpp>

// ************ Cohabitation Parallel on GPU *************************
// *******************************************************************

namespace bamboosmp {

void HybridEngine::doWorkOnGPU() {
  cudaSetDevice(0);
  cudaError_t err;

  // Create a CUDA stream
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Launch the kernel in the created stream
  ParallelGPUKernel<<<num_blocks_linear_, num_threads_per_block_, 0, stream>>>(
      n_, num_processor_, num_threads_per_block_, device_prnodes_m_,
      device_pref_lists_w_, device_partner_rank_, device_next_proposed_w_,
      device_terminate_flag_);

  // Synchronize the stream
  cudaStreamSynchronize(stream);

  // Check for errors in the kernel launch
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "CASLocalityKernel Kernel Launch err: "
              << cudaGetErrorString(err) << std::endl;
  }
  int expected = 0;
  if (atomic_host_terminate_flag_.compare_exchange_strong(expected, flag_gpu)) {
    std::cout << "bambookernel has won the contention" << std::endl;
  } else {
    std::cout << "bambookernel has lost the contention" << std::endl;
  }
}
__global__ void ParallelGPUKernel(int n, int num_processor,
                                  int num_threads_per_block, PRNode *prnodes_m,
                                  int *pref_lists_w, int *husband_rank,
                                  int *next_proposed_w, int *terminate_flag) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n) {
    // int mi_start = tid;
    int mi, mi_rank, w_idx, w_rank, mj, mj_rank;
    mi = tid;
    w_rank = 0;
    PRNode node;
    bool is_married = false;
    while (!is_married) {
      node = prnodes_m[mi * n + w_rank];
      w_idx = node.idx_;
      mi_rank = node.rank_;
      w_rank += 1;
      // if (husband_rank[w_idx] < mi_rank) {
      //   continue;
      // }
      mj_rank = atomicMin(&husband_rank[w_idx], mi_rank);
      if (mj_rank > mi_rank) {
        next_proposed_w[mi] = w_rank;
        if (mj_rank == n) {
          is_married = true;
        } else {
          mi = pref_lists_w[w_idx * n + mj_rank];
          w_rank = next_proposed_w[mi];
        }
      }
    }
  }
}

} // namespace bamboosmp