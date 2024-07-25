#pragma once

#include <atomic>
#include <queue>
#include <smp/smp.cuh>
#include <vector>

namespace bamboosmp {

class GS {
public:
  // Constructor
  GS(SMP *smp, const int &thread_limit, const int &size);
  GS(SMP *smp, std::vector<int> husband_rank_vector);
  ~GS();
  // rank matrix printer
  void PrintRankMatrix() const;
  void InitMaps();
  void InitKernelConfig(SMP *smp);
  void InitDeviceSpace();

  auto StartGS() -> float;

  auto GetMatchVector() const -> std::vector<int>;
  void PrintMatches();

private:
  SMP *smp_;
  int n_, num_blocks_, num_threads_per_block_;

  int *device_pref_lists_m_;
  int *device_pref_lists_w_;
  int *device_rank_mtx_w_;

  int *host_rank_mtx_w_;

  // queue of free men
  std::queue<int> free_men_queue_;

  int *next_proposed_w_;

  int *husband_rank_;

  std::vector<int> husband_rank_vector_;

  // parallel GS
  int num_threads_;
  std::vector<std::queue<int>> free_men_queues_;
  void initialize_queue(std::queue<int> &q, int start, int end);
  auto initialize_multiple_queues(int k, int n) -> std::vector<std::queue<int>>;
  void GSParallelProcedure(int m);
  std::atomic<int> *atomic_husband_rank_;
};

__global__ void GSInitPrefLists(int n, int num_blocks_per_list,
                                int num_threads_per_block, int *rank_mtx_m,
                                int *rank_mtx_w, int *pref_list_m,
                                int *pref_list_w);

__global__ void GSPostprocKernel(int n, int *device_partner_rank,
                                 int *device_stable_matching, int *pref_list_w);
} // namespace bamboosmp