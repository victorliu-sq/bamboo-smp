#include <smp/PRNode.cuh>

__global__ void InitPRMatrixM(int n, int *rank_mtx_w, int *pref_list_m,
                              PRNode *prnodes_m) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // row is man's idx and col is woman's idx
  PRNode node;
  if (tid < n * n) {
    int m_idx = tid / n;
    int w_rank = tid % n;

    int w_idx = pref_list_m[m_idx * n + w_rank];
    int m_rank = rank_mtx_w[w_idx * n + m_idx];

    node.idx_ = w_idx;
    node.rank_ = m_rank;
    prnodes_m[m_idx * n + w_rank] = node;
  }
}

__global__ void InitPRMatrixMCol(int n, int *rank_mtx_w, int *pref_list_m,
                                 PRNode *prnodes_m) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // row is man's idx and col is woman's idx
  PRNode node;
  if (tid < n * n) {
    int m_idx = tid / n;
    int w_rank = tid % n;

    int w_idx = pref_list_m[m_idx * n + w_rank];
    // int m_rank = rank_mtx_w[w_idx * n + m_idx];
    int m_rank = rank_mtx_w[m_idx * n + w_idx];

    node.idx_ = w_idx;
    node.rank_ = m_rank;
    prnodes_m[m_idx * n + w_rank] = node;
  }
}
