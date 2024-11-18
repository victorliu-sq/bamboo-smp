#include <smp/BambooSMP.cuh>
#include <utils/utils.hpp>

namespace bamboosmp {
    void HybridEngine::doWorkOnCPUSingleGPU() {
        const int threshold = 1;
        const int total = n_ * (n_ - 1) / 2;
        int it = 0;

        int unmached_id;
        int unmatched_num;

        cudaStream_t memcpy_stream;
        cudaStreamCreate(&memcpy_stream);

        // CUDA_CHECK(cudaSetDevice(0));
        do {
            CUDA_CHECK(cudaMemcpyAsync(temp_host_partner_rank_, device_partner_rank_,
                n_ * sizeof(int), cudaMemcpyDeviceToHost, memcpy_stream));

            CUDA_CHECK(cudaStreamSynchronize(memcpy_stream));

            unmached_id = total;
            unmatched_num = 0;
            for (int w = 0; w < n_; w++) {
                if (temp_host_partner_rank_[w] == n_) {
                    unmatched_num++;
                } else {
                    int m_rank = temp_host_partner_rank_[w];
                    unmached_id -= smp_->flatten_pref_lists_w_[w * n_ + m_rank];
                }
            }
            std::cout << "At iteration " << it << " # of unmatched men is " << unmatched_num << std::endl;

            if (unmatched_num <= threshold && unmatched_num > 0) {
                CUDA_CHECK(cudaMallocHost(&host_prnodes_m_, n_ * n_ * sizeof(PRNode),
                    cudaHostAllocDefault));

                CUDA_CHECK(cudaMemcpyAsync(host_prnodes_m_, device_prnodes_m_,
                    n_ * n_ * sizeof(PRNode), cudaMemcpyDeviceToHost, memcpy_stream));

                CUDA_CHECK(cudaStreamSynchronize(memcpy_stream));

                if (atomic_host_terminate_flag_.load() == 0) {
                    LAProcedure(unmached_id);

                    int expected = 0;
                    if (atomic_host_terminate_flag_.compare_exchange_strong(expected,
                                                                            flag_cpu)) {
                        std::cout << "CheckKernel has won the contention" << std::endl;
                        int host_terminate_flag = atomic_host_terminate_flag_.load();
                    }
                }
                unmatched_num = 0;
            }
            it++;
        } while (unmatched_num != 0);
    }
} // namespace bamboosmp
