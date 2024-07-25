#include <cstdint>
#include <iostream>
#include <smp/gs.cuh>
#include <smp/smp.cuh>
#include <thread>
#include <utils/utils.hpp>
#include <vector>

namespace bamboosmp {

GS::GS(SMP *smp, const int &thread_limit, const int &size)
    : smp_(smp), n_(size), num_threads_per_block_(thread_limit) {
  // 1. Init Kernel Config

  // 2. Init Device Space
  InitDeviceSpace();
}

void GS::InitKernelConfig(SMP *smp) {
  num_blocks_ = (n_ + num_threads_per_block_ - 1) / num_threads_per_block_;
}

void GS::InitDeviceSpace() {
  float time_copy_to_gpu, time_gpu_execution, time_copy_to_cpu,
      time_total_preprocessing;
  // host_rank_mtx_w_ = new int[n_ * n_];
  CUDA_CHECK(cudaMallocHost(&host_rank_mtx_w_, n_ * n_ * sizeof(int),
                            cudaHostAllocDefault));

  // husband_rank_ = new int[n_];
  CUDA_CHECK(
      cudaMallocHost(&husband_rank_, n_ * sizeof(int), cudaHostAllocDefault));

  next_proposed_w_ = new int[n_];

  atomic_husband_rank_ = new std::atomic<int>[n_];

  // CUDA_CHECK(cudaMalloc((void **)&device_pref_lists_m_, n_ * n_ *
  // sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&device_pref_lists_w_, n_ * n_ * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&device_rank_mtx_w_, n_ * n_ * sizeof(int)));

  CUDA_CHECK(cudaDeviceSynchronize());

  cudaError_t err;
  uint64_t start_memcpy, end_memcpy;
  start_memcpy = getNanoSecond();
  CUDA_CHECK(cudaMemcpy(device_pref_lists_w_, smp_->flatten_pref_lists_w_,
                        n_ * n_ * sizeof(int), cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  end_memcpy = getNanoSecond();
  time_copy_to_gpu = (end_memcpy - start_memcpy) * 1.0 / 1e6;
  std::cout << "GS Copy PrefLists into GPU spends " << time_copy_to_gpu
            << " ms " << std::endl;

  // p >= n^2
  int threadsPerBlock = num_threads_per_block_;

  int init_num_blocks = (n_ * n_ + threadsPerBlock - 1) / threadsPerBlock;
  // printf("Initialization: Launch %d blocks\n", init_num_blocks);

  auto start_init_nodes = getNanoSecond();

  InitRankMatrix<<<init_num_blocks, num_threads_per_block_>>>(
      n_, init_num_blocks, num_threads_per_block_, device_rank_mtx_w_,
      device_pref_lists_w_);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "GSInit Launch err: " << cudaGetErrorString(err) << std::endl;
  }
  cudaDeviceSynchronize();

  auto end_init_nodes = getNanoSecond();

  time_gpu_execution = (end_init_nodes - start_init_nodes) * 1.0 / 1e6;
  std::cout << "GS Init RankMatrixW  in parallel on GPU spends "
            << time_gpu_execution << " ms" << std::endl;

  // Copy back to CPU
  start_memcpy = getNanoSecond();
  CUDA_CHECK(cudaMemcpy(host_rank_mtx_w_, device_rank_mtx_w_,
                        n_ * n_ * sizeof(int), cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize(); // Ensure prefetching is complete
  end_memcpy = getNanoSecond();
  time_copy_to_cpu = (end_memcpy - start_memcpy) * 1.0 / 1e6;
  std::cout << "GS Copy back RankMatrixW to CPU spends " << time_copy_to_cpu
            << " ms " << std::endl;

  time_total_preprocessing =
      time_copy_to_gpu + time_copy_to_cpu + time_gpu_execution;

  std::cout << "Total GS preprocessing time is " << time_total_preprocessing
            << " ms " << std::endl;

  auto start_time = getNanoSecond();

  // std::cout << "Init husband_rank starts" << std::endl;
  for (int i = 0; i < n_; ++i) {
    free_men_queue_.push(i);
    husband_rank_[i] = n_;

    // std::cout << "husband_rank[" << i << "] is " << husband_rank_[i]
    //           << std::endl;
    next_proposed_w_[i] = 0;
    atomic_husband_rank_[i].store(n_);
  }
  // std::cout << "Init husband_rank is done" << std::endl;

  auto end_time = getNanoSecond();
  auto time = (end_time - start_time) * 1.0 / 1e6;
  std::cout << "GS init other data structures require " << time << " ms "
            << std::endl;

  num_threads_ =
      std::thread::hardware_concurrency(); // Get the number of available
                                           // hardware threads
  // num_threads_ = 12;

  if (num_threads_ > n_) {
    num_threads_ = n_;
  }

  // std::cout << "Total number of threads is " << num_threads_ << std::endl;

  // initialize queues for parallel GS
  free_men_queues_ = initialize_multiple_queues(num_threads_, n_);
}

// Function to initialize a single queue
void GS::initialize_queue(std::queue<int> &q, int start, int end) {
  // printf("man %d ~ %d have been pushed into queue\n", start, end - 1);
  for (int i = start; i < end; ++i) {
    q.push(i);
  }
}

// Function to initialize multiple queues in parallel
auto GS::initialize_multiple_queues(int k, int n)
    -> std::vector<std::queue<int>> {
  std::vector<std::queue<int>> queues(k);
  std::vector<std::thread> threads;

  int avg_num_men = n / k;

  // Launch threads
  for (int i = 0; i < k; ++i) {
    if (i != k - 1) {
      threads.emplace_back(&GS::initialize_queue, this, std::ref(queues[i]),
                           i * avg_num_men, (i + 1) * avg_num_men);
    } else {
      threads.emplace_back(&GS::initialize_queue, this, std::ref(queues[i]),
                           i * avg_num_men, n);
    }
  }

  // Join threads
  for (auto &th : threads) {
    th.join();
  }

  return queues;
}

__global__ void GSInitPrefLists(int n, int num_blocks_per_list,
                                int num_threads_per_block, int *rank_mtx_m,
                                int *rank_mtx_w, int *pref_list_m,
                                int *pref_list_w) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // row is man's idx and col is woman's idx
  if (tid < n * n) {
    int m_idx = tid / n;
    int w_idx = tid % n;

    int w_rank = rank_mtx_m[m_idx * n + w_idx];
    int m_rank = rank_mtx_w[w_idx * n + m_idx];

    pref_list_w[w_idx * n + m_rank] = m_idx;
    pref_list_m[m_idx * n + w_rank] = w_idx;
  }
}

GS::~GS() {
  // delete[] husband_rank_;
  delete[] next_proposed_w_;
  CUDA_CHECK(cudaFree(device_pref_lists_w_));
  // CUDA_CHECK(cudaFree(device_pref_lists_m_));
  CUDA_CHECK(cudaFree(device_rank_mtx_w_));

  // delete[] host_rank_mtx_w_;
  CUDA_CHECK(cudaFreeHost(host_rank_mtx_w_));
  // cudaDeviceReset()
  // cudaDeviceSynchronize();
}

} // namespace bamboosmp