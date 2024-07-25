#include <iostream>
#include <smp/BambooSMP.cuh>
#include <smp/flatten.cuh>
#include <smp/gs.cuh>
#include <smp/smp.cuh>
#include <utils/generateWorkloadCongested.hpp>
#include <utils/generateWorkloadPerfect.hpp>
#include <utils/generateWorkloadRandom.hpp>
#include <utils/generateWorkloadSolo.hpp>
#include <utils/utils.hpp>

int main() {
  int n = 5;
  int thread_limit = 128;
  bamboosmp::SMP *smp;

  bamboosmp::HybridEngine::CheckAndSetupCudaDevices();
  std::string filepath;
  char input_char;

  bool isDone = false;
  while (!isDone) {
    std::cout << "Enter filepath (or 'q' to quit): ";
    std::cin >> filepath;

    if (filepath == "q") {
      isDone = true;
    } else {
      smp = bamboosmp::readPrefListsFromJson(filepath);
      auto bamboosmp = new bamboosmp::HybridEngine(smp, thread_limit, n);
      bamboosmp->Solve();
      bamboosmp->PrintMatching();
    }
  }

  return 0;
}