#pragma once

#include <vector>

struct PRNode {
  int idx_;
  int rank_;
};

using PRMatrix = std::vector<std::vector<PRNode>>;

__global__ void InitPRMatrixM(int n, int *rank_mtx_w, int *pref_list_m,
                              PRNode *prnodes_m);

__global__ void InitPRMatrixMCol(int n, int *rank_mtx_w, int *pref_list_m,
                                 PRNode *prnodes_m);