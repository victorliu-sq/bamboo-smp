#include <smp/BambooSMP.cuh>
#include <utils/utils.hpp>

namespace bamboosmp {
    void HybridEngine::doWorkOnCPU() {
        int unmatched_id;
        int unmatched_num;

        cudaStream_t memcpy_stream;
        cudaStreamCreate(&memcpy_stream);

        MonitorProceture(unmatched_id, unmatched_num, memcpy_stream);

        if (unmatched_num == 1) {
            CUDA_CHECK(cudaMallocHost(&host_prnodes_m_, n_ * n_ * sizeof(PRNode),
                cudaHostAllocDefault));

            CUDA_CHECK(cudaMemcpyAsync(host_prnodes_m_, device_prnodes_m_,
                n_ * n_ * sizeof(PRNode), cudaMemcpyDeviceToHost, memcpy_stream));

            CUDA_CHECK(cudaStreamSynchronize(memcpy_stream));

            if (atomic_host_terminate_flag_.load() == 0) {
                LAProcedure(unmatched_id);

                int expected = 0;
                if (atomic_host_terminate_flag_.compare_exchange_strong(expected,
                                                                        flag_cpu)) {
                    std::cout << "CheckKernel has won the contention" << std::endl;
                    int host_terminate_flag = atomic_host_terminate_flag_.load();
                }
            }
            unmatched_num = 0;
        }
    }

    void HybridEngine::MonitorProceture(int &unmatched_id, int &unmatched_num, cudaStream_t &stream) {
        int it = 0;
        const int total = n_ * (n_ - 1) / 2;
        do {
            CUDA_CHECK(cudaMemcpyAsync(temp_host_partner_rank_, device_partner_rank_,
                n_ * sizeof(int), cudaMemcpyDeviceToHost, stream));

            CUDA_CHECK(cudaStreamSynchronize(stream));

            unmatched_id = total;
            unmatched_num = 0;
            for (int w = 0; w < n_; w++) {
                if (temp_host_partner_rank_[w] == n_) {
                    unmatched_num++;
                } else {
                    int m_rank = temp_host_partner_rank_[w];
                    unmatched_id -= smp_->flatten_pref_lists_w_[w * n_ + m_rank];
                }
            }
            std::cout << "At iteration " << it << " # of unmatched men is " << unmatched_num << std::endl;

            it++;
        } while (unmatched_num > 1);
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
} // namespace bamboosmp
