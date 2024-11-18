#include "smp/smp.cuh"
#include <smp/gs.cuh>
#include <utils/utils.hpp>

namespace bamboosmp {
  auto GS::StartGS() -> float {
    auto start_time = getNanoSecond();
    int m_idx, m_rank, w_idx, w_rank, p_rank;

    m_idx = free_men_queue_.front();
    free_men_queue_.pop();
    w_rank = 0;
    bool done = false;
    int iteration = 0;
    while (!done) {
      iteration += 1;
      w_idx = smp_->flatten_pref_lists_m_[m_idx * n_ + w_rank];
      m_rank = host_rank_mtx_w_[w_idx * n_ + m_idx];
      // std::cout << "Man " << m_idx << " proposed to " << w_rank << " th Woman "
      //     << w_idx;
      w_rank += 1;

      p_rank = husband_rank_[w_idx];
      // std::cout << "p_rank: " << p_rank << "; m_rank: " << m_rank << std::endl;

      if (m_rank < p_rank) {
        // std::cout << " And succeeds!" << std::endl;
        husband_rank_[w_idx] = m_rank;
        if (p_rank != n_) {
          int new_free_man = smp_->flatten_pref_lists_w_[w_idx * n_ + p_rank];
          free_men_queue_.push(new_free_man);
          // std::cout << "man " << new_free_man << " becomes free again"
          //     << std::endl;
        }
        next_proposed_w_[m_idx] = w_rank;
        if (!free_men_queue_.empty()) {
          m_idx = free_men_queue_.front();
          free_men_queue_.pop();
          w_rank = next_proposed_w_[m_idx];
        } else {
          done = true;
        }
      } else {
        // std::cout << " And fails" << std::endl;
      }
    }
    // std::cout << "number of iterations: " << iteration << std::endl;
    auto end_time = getNanoSecond();

    return (end_time - start_time) / 1e6;
  }

  __global__ void GSPostprocKernel(int n, int *device_partner_rank,
                                   int *device_stable_matching,
                                   int *pref_list_w) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
      int w = tid;
      int p_rank = device_partner_rank[w];
      int m = pref_list_w[w * n + p_rank];
      device_stable_matching[m] = w;
    }
  }

  auto GS::GetMatchVector() const -> std::vector<int> {
    std::vector<int> match_vec(n_);

    auto start_post = getNanoSecond();
    int *device_match;
    int *device_partner_rank;
    cudaMalloc(&device_match, n_ * sizeof(int));
    cudaMalloc(&device_partner_rank, n_ * sizeof(int));
    cudaMemcpy(device_partner_rank, husband_rank_, n_ * sizeof(int),
               cudaMemcpyHostToDevice);

    int threadsPerBlock = num_threads_per_block_;

    int init_num_blocks = (n_ + threadsPerBlock - 1) / threadsPerBlock;
    GSPostprocKernel<<<init_num_blocks, threadsPerBlock>>>(
      n_, device_partner_rank, device_match, device_pref_lists_w_);

    cudaMemcpy(match_vec.data(), device_match, n_ * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_match);

    // for (int w_id = 0; w_id < match_vec.size(); w_id++) {
    //   int m_rank = husband_rank_[w_id];
    //   int m_id = smp_->flatten_pref_lists_w_[w_id * n_ + m_rank];
    //   match_vec[m_id] = w_id;
    // }

    auto end_post = getNanoSecond();
    float post_time = (end_post - start_post) * 1.0 / 1e6;
    std::cout << "Postprocess takes " << post_time << " milliseconds"
        << std::endl;
    return match_vec;
  }

  void GS::PrintMatches() {
    auto match_vec_row = GetMatchVector();
    std::cout << "*************** Matchings Start ***************" << std::endl;
    for (int i = 0; i < match_vec_row.size(); i++) {
      std::cout << "Matching " << i << " with " << match_vec_row[i] << std::endl;
    }
    std::cout << "*************** Matchings   End ***************" << std::endl;
  }
} // namespace bamboosmp
