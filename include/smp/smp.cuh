#pragma once
#include <smp/PRNode.cuh>
#include <string>
#include <utils/utils.hpp>
#include <vector>

namespace bamboosmp {
    class SMP {
    public:
        SMP(const PreferenceLists &plM, const PreferenceLists &plW, const int& n);

        ~SMP();

        int *flatten_pref_lists_m_;
        int *flatten_pref_lists_w_;
    };

    auto readPrefListsFromJson(const std::string &filepath) -> SMP *;

    __global__ void InitRankMatrix(int n, int num_blocks_per_list,
                                   int num_threads_per_block, int *rank_mtx,
                                   int *pref_list);

    __global__ void InitRankMatrixCol(int n, int num_blocks_per_list,
                                      int num_threads_per_block, int *rank_mtx,
                                      int *pref_list);
} // namespace bamboosmp
