#include <iostream>
#include <smp/BambooSMP.cuh>
#include <thread>
#include <utils/utils.hpp>

// ********************* Cohabitation Hybrid *************************
// *******************************************************************
namespace bamboosmp {
  void HybridEngine::Exec() {
    if (!is_perfect_) {
      std::thread CPUThread(&HybridEngine::doWorkOnCPU, this);
      std::thread GPUThread(&HybridEngine::doWorkOnGPU, this);

      int mode = 0;

      auto sleepDuration = std::chrono::microseconds(1);

      do {
        std::this_thread::sleep_for(sleepDuration);
        mode = atomic_host_terminate_flag_.load();
        if (mode == flag_cpu) {
          std::cout << "Bamboo: Thread CheckKernel is done" << std::endl;
          CPUThread.join();
          GPUThread.detach();
        } else if (mode == flag_gpu) {
          std::cout << "Bamboo: Thread BambooKernel is done" << std::endl;
          GPUThread.join();
          CPUThread.detach();
        }
      } while (mode == 0);
    }
  }

  void HybridEngine::ExecSingleGPU() {
    if (!is_perfect_) {
      std::thread CPUThread(&HybridEngine::doWorkOnCPUSingleGPU, this);
      std::thread GPUThread(&HybridEngine::doWorkOnGPU, this);

      int mode = 0;

      auto sleepDuration = std::chrono::microseconds(1);

      do {
        std::this_thread::sleep_for(sleepDuration);
        mode = atomic_host_terminate_flag_.load();
        if (mode == flag_cpu) {
          std::cout << "Bamboo: Thread CheckKernel is done" << std::endl;
          CPUThread.join();
          GPUThread.detach();
        } else if (mode == flag_gpu) {
          std::cout << "Bamboo: Thread BambooKernel is done" << std::endl;
          GPUThread.join();
          CPUThread.detach();
        }
      } while (mode == 0);
    }
    return;
  }
} // namespace bamboosmp
