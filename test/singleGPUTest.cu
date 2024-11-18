//
// Created by victor on 11/17/24.
//


#include "testbase.h"
#include <memory>
#include <smp/gs.cuh>

#include "../include/smp/flatten.cuh"
#include "../include/smp/BambooSMP.cuh"
#include "../include/utils/generateWorkloadRandom.hpp"


void test_single_gpu_random_5(TestContext &ctx) {
  const int n = 5;
  const int group_size = 3;
  const int thread_limit = 5;
  const int num_processor = 5;
  auto plM = GeneratePrefListsRandom(n, group_size);
  auto plW = GeneratePrefListsRandom(n, group_size);

  auto flatten_pref_lists_m = ParallelFlattenHost(plM, n);
  auto flatten_pref_lists_w = ParallelFlattenHost(plW, n);

  auto smp = make_shared<bamboosmp::SMP>(flatten_pref_lists_m, flatten_pref_lists_w);
  auto gs = make_shared<bamboosmp::GS>(smp.get(), 5, 5);
  auto time_gs = gs->StartGS();
  auto match_vec_gs = gs->GetMatchVector();

  auto bamboosmp =
      new bamboosmp::HybridEngine(smp.get(), thread_limit, num_processor);
  bamboosmp->SolveSingleGPU();
  auto match_vec_la = bamboosmp->GetStableMatching();

  // Check if vectors are equal
  bool correct = true;
  for (size_t i = 0; i < n; i++) {
    // EXPECT_EQ(match_vec_la[i], match_vec_gs[i]);
    if (match_vec_la[i] != match_vec_gs[i]) {
      correct = false;
    }
  }
  if (correct) {
    std::cout << "Correct !" << std::endl;
  } else {
    std::cout << "Wrong :(" << std::endl;
  }
  std::cout << "GS time is: " << time_gs << std::endl;
}

int main() {
  cout << "Testing the TreeSet class." << endl << endl;

  TestContext ctx(cout);

  test_single_gpu_random_5(ctx);

  // Return 0 if everything passed, nonzero if something failed.
  return !ctx.ok();
}

