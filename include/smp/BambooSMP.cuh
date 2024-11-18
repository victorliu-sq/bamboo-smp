#pragma once

#include "smp.cuh"
#include <atomic>
#include <smp/PRNode.cuh>
#include <vector>

namespace bamboosmp {
    class HybridEngine {
    public:
        // Constructor
        HybridEngine(SMP *smp, const int &thread_limit, const int &size);

        ~HybridEngine();

        void Solve();

        void SolveSingleGPU();

        auto GetStableMatching() const -> std::vector<int>;

        void PrintMatching() const;

        static void CheckAndSetupCudaDevices();

    private:
        // Kernel Config
        void Precheck();

        void Init();

        void Exec();

        void ExecSingleGPU();

        void Postproc();

        static const int flag_cpu;
        static const int flag_gpu;

        SMP *smp_;
        int num_threads_;
        int mode_;

        // Prechecking
        bool is_perfect_;
        // std::vector<int> stable_matching_;
        int *stable_matching_;

        int n_, num_threads_per_block_, num_processor_;
        int num_blocks_linear_, num_blocks_square_;

        int *device_pref_lists_m_;
        int *managed_pref_lists_m_;

        int *device_pref_lists_w_;
        int *device_rank_mtx_w_;
        int *host_rank_mtx_w_;

        PRNode *device_prnodes_m_;
        PRNode *host_prnodes_m_;

        // for sequential algorithm
        void LAProcedure(int m);

        // for parallel algorithm on GPU
        int *host_partner_rank_;
        int *temp_host_partner_rank_;
        int *device_partner_rank_;
        int *split_device_husband_rank_;

        int *device_free_man_idx_;
        int *device_num_unproposed_;

        int *host_next_proposed_w_;
        int *split_device_next_proposed_w_;
        int *device_next_proposed_w_;

        int *device_terminate_flag_;
        // int *host_terminate_flag_;
        std::atomic<int> atomic_host_terminate_flag_;

        void doWorkOnGPU();

        void doWorkOnCPU();

        void doWorkOnCPUSingleGPU();

        void InitPRMatrixProcedure();
    };

    __global__ void ParallelGPUKernel(int n, int num_processor,
                                      int num_threads_per_block, PRNode *prnodes_m,
                                      int *pref_lists_w, int *husband_rank,
                                      int *next_proposed_w, int *termiante_flag);

    __global__ void IdentifyUnmatchedMan(int n, int *num_unproposed,
                                         int *husband_rank, int *split_husband_rank,
                                         int *free_man_idx, int *pref_lists_w,
                                         int *next, int *split_next);
} // namespace bamboosmp
