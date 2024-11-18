#include <atomic>
#include <iostream>
#include <smp/BambooSMP.cuh>
#include <smp/PRNode.cuh>
#include <utils/utils.hpp>

namespace bamboosmp {
    void HybridEngine::Init() {
        cudaError_t err;

        num_blocks_linear_ =
                (n_ + num_threads_per_block_ - 1) / num_threads_per_block_;

        num_blocks_square_ =
                (n_ * n_ + num_threads_per_block_ - 1) / num_threads_per_block_;

        CUDA_CHECK(cudaSetDevice(0));

        CUDA_CHECK(cudaMallocHost(&host_next_proposed_w_, n_ * sizeof(int),
            cudaHostAllocDefault));
        CUDA_CHECK(cudaMallocHost(&host_partner_rank_, n_ * sizeof(int),
            cudaHostAllocDefault));
        CUDA_CHECK(cudaMallocHost(&temp_host_partner_rank_, n_ * sizeof(int),
            cudaHostAllocDefault));

        CUDA_CHECK(cudaMallocHost(&host_rank_mtx_w_, n_*n_ * sizeof(int),
            cudaHostAllocDefault));
        std::atomic_init(&atomic_host_terminate_flag_, 0);

        CUDA_CHECK(cudaMalloc((void **)&device_pref_lists_m_, n_ * n_ * sizeof(int)));

        CUDA_CHECK(cudaMalloc((void **)&device_pref_lists_w_, n_ * n_ * sizeof(int)));

        CUDA_CHECK(cudaMalloc((void **)&device_rank_mtx_w_, n_ * n_ * sizeof(int)));

        CUDA_CHECK(cudaMalloc((void **)&device_prnodes_m_, n_ * n_ * sizeof(PRNode)));

        CUDA_CHECK(cudaMalloc((void **)&device_partner_rank_, n_ * sizeof(int)));

        CUDA_CHECK(cudaMalloc((void **)&device_next_proposed_w_, n_ * sizeof(int)));

        CUDA_CHECK(cudaMalloc((void **)&device_terminate_flag_, sizeof(int)));

        int host_terminate_flag = 0;
        CUDA_CHECK(cudaMemcpy(device_terminate_flag_, &host_terminate_flag,
            sizeof(int), cudaMemcpyHostToDevice));

        cudaDeviceSynchronize();

        CUDA_CHECK(cudaMemcpy(device_pref_lists_m_, smp_->flatten_pref_lists_m_,
            n_ * n_ * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(device_pref_lists_w_, smp_->flatten_pref_lists_w_,
            n_ * n_ * sizeof(int), cudaMemcpyHostToDevice));

        cudaDeviceSynchronize();

        cudaDeviceSynchronize();
        auto start_init_nodes = getNanoSecond();

        InitPRMatrixProcedure();

        auto start_others = getNanoSecond();
        for (int i = 0; i < n_; ++i) {
            host_partner_rank_[i] = n_;
            host_next_proposed_w_[i] = 0;
        }

        cudaMemcpy(device_partner_rank_, host_partner_rank_, n_ * sizeof(int),
                   cudaMemcpyHostToDevice);

        CUDA_CHECK(cudaMemcpy(device_next_proposed_w_, host_next_proposed_w_,
            n_ * sizeof(int), cudaMemcpyHostToDevice));

        // disable this for sinlgeGPU
        // CUDA_CHECK(cudaSetDevice(1));
        // CUDA_CHECK(cudaMalloc((void **)&device_num_unproposed_, sizeof(int)));
        // CUDA_CHECK(cudaMalloc(&device_free_man_idx_, sizeof(int)));
        // CUDA_CHECK(cudaMalloc(&split_device_husband_rank_, n_ * sizeof(int)));
        // CUDA_CHECK(cudaMalloc(&split_device_next_proposed_w_, n_ * sizeof(int)));

        // CUDA_CHECK(cudaMemcpy(split_device_husband_rank_, host_partner_rank_, n_ * sizeof(int),
        //     cudaMemcpyHostToDevice));
    }

    void HybridEngine::InitPRMatrixProcedure() {
        InitRankMatrixCol<<<num_blocks_square_, num_threads_per_block_>>>(
            n_, num_blocks_square_, num_threads_per_block_, device_rank_mtx_w_,
            device_pref_lists_w_);

        cudaDeviceSynchronize();

        InitPRMatrixMCol<<<num_blocks_square_, num_threads_per_block_>>>(
            n_, device_rank_mtx_w_, device_pref_lists_m_, device_prnodes_m_);
        cudaDeviceSynchronize();
    }
} // namespace bamboosmp
