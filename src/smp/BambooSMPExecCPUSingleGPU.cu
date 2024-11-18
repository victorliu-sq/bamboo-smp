#include <smp/BambooSMP.cuh>
#include <utils/utils.hpp>

namespace bamboosmp {
    void HybridEngine::doWorkOnCPUSingleGPU() {
        const int threshold = 1;
        const int total = n_ * (n_ - 1) / 2;
        int it = 0;

        int host_free_man_idx;
        int host_num_unproposed;

        // CUDA_CHECK(cudaSetDevice(0));

        CUDA_CHECK(cudaMemcpy(host_rank_mtx_w_, device_rank_mtx_w_,
            n_ * n_ * sizeof(int), cudaMemcpyDeviceToHost));

        do {
            host_free_man_idx = total;
            host_num_unproposed = 0;

            CUDA_CHECK(cudaMemcpy(host_partner_rank_, device_partner_rank_,
                n_ * sizeof(int), cudaMemcpyDeviceToHost));

            for (int w = 0; w < n_; w++) {
                if (host_partner_rank_[w] == n_) {
                    host_num_unproposed++;
                } else {
                    host_free_man_idx -= host_rank_mtx_w_[host_partner_rank_[w * n_ + host_partner_rank_[w]]];
                }
            }

            if (host_num_unproposed <= threshold && host_num_unproposed > 0) {
                CUDA_CHECK(cudaMallocHost(&host_prnodes_m_, n_ * n_ * sizeof(PRNode),
                    cudaHostAllocDefault));

                CUDA_CHECK(cudaMemcpy(host_prnodes_m_, device_prnodes_m_,
                    n_ * n_ * sizeof(PRNode), cudaMemcpyDeviceToHost));

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
    }
} // namespace bamboosmp
