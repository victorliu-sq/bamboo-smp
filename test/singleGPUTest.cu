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

// random 0, congested 1, solo 2
#define WORKLOAD_RANDOM 0
#define WORKLOAD_CONGESTED 1
#define WORKLOAD_SOLO 2
#define WORKLOAD_TYPE 2

void test_single_gpu(TestContext &ctx) {
  int workload_type;

  const int n = 10000;
  const int group_size = 5;
  const int thread_per_block = 128;

  PreferenceLists plM;
  PreferenceLists plW;
  if (WORKLOAD_TYPE == WORKLOAD_RANDOM) {
    plM = GeneratePrefListsRandom(n, group_size);
    plW = GeneratePrefListsRandom(n, group_size);
  } else if (WORKLOAD_TYPE == WORKLOAD_CONGESTED) {
    plM = GeneratePrefListsCongested(n);
    plW = GeneratePrefListsCongested(n);
  } else {
    plM = GeneratePrefListsManSolo(n);
    plW = GeneratePrefListsWomanSolo(n);
  }

  auto smp = new bamboosmp::SMP(plM, plW, n);
  auto gs = new bamboosmp::GS(smp, thread_per_block, n);
  auto time_gs = gs->StartGS();
  auto match_vec_gs = gs->GetMatchVector();

  delete gs;

  auto bamboosmp =
      new bamboosmp::HybridEngine(smp, thread_per_block, n);
  bamboosmp->SolveSingleGPU();
  auto start_time_bamboo = getNanoSecond();
  auto match_vec_la = bamboosmp->GetStableMatching();
  auto end_time_bamboo = getNanoSecond();
  auto time_bamboo = (end_time_bamboo - start_time_bamboo) / 1e6;

  // Check if vectors are equal
  bool correct = true;

  for (size_t i = 0; i < n; i++) {
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

