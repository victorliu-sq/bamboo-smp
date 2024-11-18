//
// Created by victor on 11/17/24.
//


#include "testbase.h"
#include <memory>
#include <smp/gs.cuh>
#include <utils/generateWorkloadCongested.hpp>

#include "../include/smp/flatten.cuh"
#include "../include/smp/BambooSMP.cuh"
#include "../include/utils/generateWorkloadRandom.hpp"
#include "../include/utils/generateWorkloadSolo.hpp"


void test_single_gpu(TestContext &ctx) {
  int workload_type;
  const int mode_random = 0;
  const int mode_congested = 1;
  const int mode_clustered = 2;
  const int mode_solo = 3;

  const int n = 10000;
  const int group_size = 5;
  const int thread_limit = 128;
  const int num_processor = n;

  // workload_type = mode_random;
  // workload_type = mode_congested;
  workload_type = mode_solo;

  PreferenceLists plM;
  PreferenceLists plW;
  if (workload_type == mode_random) {
    plM = GeneratePrefListsRandom(n, group_size);
    plW = GeneratePrefListsRandom(n, group_size);
  } else if (workload_type == mode_congested) {
    plM = GeneratePrefListsCongested(n);
    plW = GeneratePrefListsCongested(n);
  } else {
    plM = GeneratePrefListsManSolo(n);
    plW = GeneratePrefListsWomanSolo(n);
  }

  auto flatten_pref_lists_m = ParallelFlattenHost(plM, n);
  auto flatten_pref_lists_w = ParallelFlattenHost(plW, n);

  auto smp = new bamboosmp::SMP(flatten_pref_lists_m, flatten_pref_lists_w);
  auto gs = new bamboosmp::GS(smp, thread_limit, num_processor);
  auto time_gs = gs->StartGS();
  auto match_vec_gs = gs->GetMatchVector();
  // std::cout << "After GS:" << std::endl;
  // for (int m = 0; m < n; ++m) {
  //   std::cout << "man " << m << " is paired with woman " << match_vec_gs[m] << std::endl;
  // }
  // std::cout << std::endl;
  delete gs;

  auto bamboosmp =
      new bamboosmp::HybridEngine(smp, thread_limit, num_processor);
  bamboosmp->SolveSingleGPU();
  auto start_time_bamboo = getNanoSecond();
  auto match_vec_la = bamboosmp->GetStableMatching();
  auto end_time_bamboo = getNanoSecond();
  auto time_bamboo = (end_time_bamboo - start_time_bamboo) / 1e6;

  // Check if vectors are equal
  bool correct = true;
  // std::cout << "After BambooSMP:" << std::endl;
  // for (int m = 0; m < n; ++m) {
  //   std::cout << "man " << m << " is paired with woman " << match_vec_la[m] << std::endl;
  // }
  // std::cout << std::endl;


  for (size_t i = 0; i < n; i++) {
    // EXPECT_EQ(match_vec_la[i], match_vec_gs[i]);
    if (match_vec_la[i] != match_vec_gs[i]) {
      correct = false;
      std::cout << "man " << i << " is paired with woman " << match_vec_la[i] << "in BambooSMP " << std::endl;
      std::cout << "man " << i << " is paired with woman " << match_vec_gs[i] << "in GS " << std::endl;
    }
  }
  if (correct) {
    std::cout << "Correct !" << std::endl;
  } else {
    std::cout << "Wrong :(" << std::endl;
  }
  std::cout << "GS time is: " << time_gs << std::endl;
  std::cout << "Bamboo time is: " << time_bamboo << std::endl;
}

int main() {
  cout << "Testing the TreeSet class." << endl << endl;

  TestContext ctx(cout);

  test_single_gpu(ctx);

  // Return 0 if everything passed, nonzero if something failed.
  return !ctx.ok();
}

