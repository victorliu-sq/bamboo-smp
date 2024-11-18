#include <smp/BambooSMP.cuh>
#include <utils/utils.hpp>

namespace bamboosmp {
  void HybridEngine::doWorkOnCPU() {
    int mode;
    int host_free_man_idx;
    int total = n_ * (n_ - 1) / 2;
    int host_num_unproposed;
    int it = 0;
    int threshold = 1;
    int num_blocks = (n_ + num_threads_per_block_ - 1) / num_threads_per_block_;

    CUDA_CHECK(cudaSetDevice(1));
    cudaError_t err;

    // Create a CUDA stream
    cudaStream_t memcpy_stream;
    cudaStreamCreate(&memcpy_stream);

    do {
      host_free_man_idx = total;
      host_num_unproposed = 0;

      CUDA_CHECK(cudaMemcpyAsync(device_num_unproposed_, &host_num_unproposed,
        sizeof(int), cudaMemcpyHostToDevice,
        memcpy_stream));
      CUDA_CHECK(cudaMemcpyAsync(device_free_man_idx_, &host_free_man_idx,
        sizeof(int), cudaMemcpyHostToDevice,
        memcpy_stream));

      IdentifyUnmatchedMan<<<num_blocks, num_threads_per_block_, 0,
          memcpy_stream>>>(
            n_, device_num_unproposed_, device_partner_rank_,
            split_device_husband_rank_, device_free_man_idx_, device_pref_lists_w_,
            device_next_proposed_w_, split_device_next_proposed_w_);

      err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::cout << "checkLessThanN Kernel Launch err: "
            << cudaGetErrorString(err) << std::endl;
      }

      CUDA_CHECK(cudaMemcpyAsync(&host_num_unproposed, device_num_unproposed_,
        sizeof(int), cudaMemcpyDeviceToHost,
        memcpy_stream));

      // Synchronize the stream to ensure all memcpy and kernel operations are
      // completed
      CUDA_CHECK(cudaStreamSynchronize(memcpy_stream));

      if (host_num_unproposed <= threshold && host_num_unproposed > 0) {
        CUDA_CHECK(cudaMallocHost(&host_prnodes_m_, n_ * n_ * sizeof(PRNode),
          cudaHostAllocDefault));

        CUDA_CHECK(cudaMemcpyAsync(&host_free_man_idx, device_free_man_idx_,
          sizeof(int), cudaMemcpyDeviceToHost,
          memcpy_stream));

        CUDA_CHECK(cudaMemcpyAsync(host_partner_rank_, split_device_husband_rank_,
          n_ * sizeof(int), cudaMemcpyDeviceToHost,
          memcpy_stream));

        cudaMemcpyAsync(host_prnodes_m_, device_prnodes_m_,
                        n_ * n_ * sizeof(PRNode), cudaMemcpyDeviceToHost,
                        memcpy_stream);

        // Synchronize the stream to ensure all memcpy operations are completed
        CUDA_CHECK(cudaStreamSynchronize(memcpy_stream));

        if (atomic_host_terminate_flag_.load() == 0) {
          LAProcedure(host_free_man_idx);

          int expected = 0;
          if (atomic_host_terminate_flag_.compare_exchange_strong(expected,
                                                                  flag_cpu)) {
            std::cout << "CheckKernel has won the contention" << std::endl;
            int host_terminate_flag = atomic_host_terminate_flag_.load();
          }
        }
        host_num_unproposed = 0;
      }
      it++;
    } while (host_num_unproposed != 0);

    // Destroy the stream
    CUDA_CHECK(cudaStreamDestroy(memcpy_stream));
  }

  void HybridEngine::LAProcedure(int m) {
    for (int w = 0; w < n_; w++) {
      host_partner_rank_[w] = temp_host_partner_rank_[w];
    }

    int w_idx, m_rank, m_idx, w_rank, p_rank;
    m_idx = m;
    w_rank = 0;
    PRNode temp_node;
    bool is_matched = false;
    int iterations = 0;
    while (!is_matched) {
      iterations += 1;
      temp_node = host_prnodes_m_[m_idx * n_ + w_rank];
      w_idx = temp_node.idx_;
      m_rank = temp_node.rank_;
      p_rank = host_partner_rank_[w_idx];
      if (p_rank == n_) {
        host_next_proposed_w_[m_idx] = w_rank;
        host_partner_rank_[w_idx] = m_rank;
        is_matched = true;
      } else if (p_rank > m_rank) {
        host_next_proposed_w_[m_idx] = w_rank;
        host_partner_rank_[w_idx] = m_rank;

        m_idx = smp_->flatten_pref_lists_w_[w_idx * n_ + p_rank];
        w_rank = host_next_proposed_w_[m_idx];
      } else {
        w_rank++;
      }
    }
  }

  __global__ void IdentifyUnmatchedMan(int n, int *num_unproposed,
                                       int *partner_rank, int *split_partner_rank,
                                       int *free_man_idx, int *pref_lists_w,
                                       int *next, int *split_next) {
    int wi = blockIdx.x * blockDim.x + threadIdx.x;

    if (wi < n) {
      split_next[wi] = next[wi];

      int hr = partner_rank[wi];
      split_partner_rank[wi] = hr;
      if (hr == n) {
        atomicAdd(num_unproposed, 1);
      } else {
        atomicSub(free_man_idx, pref_lists_w[wi * n + hr]);
      }
    }
  }
} // namespace bamboosmp
