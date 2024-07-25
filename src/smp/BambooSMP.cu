#include <smp/BambooSMP.cuh>
#include <smp/PRNode.cuh>
#include <utils/utils.hpp>
#include <vector>

namespace bamboosmp {

const int HybridEngine::flag_cpu = 1;
const int HybridEngine::flag_gpu = 2;

HybridEngine::HybridEngine(SMP *smp, const int &thread_limit, const int &size)
    : smp_(smp), n_(size), num_threads_per_block_(thread_limit) {}

// Warm-up kernel function
__global__ void warmUpKernel() {}

void HybridEngine::CheckAndSetupCudaDevices() {
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  if (error_id != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount returned " << static_cast<int>(error_id)
              << " -> " << cudaGetErrorString(error_id) << std::endl;
    std::cerr << "Result = FAIL" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (deviceCount < 2) {
    std::cerr << "At least 2 CUDA devices are required. Found " << deviceCount
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // Set access between device 0 and device 1
  // cudaSetDevice(0);
  // cudaDeviceEnablePeerAccess(1, 0);
  cudaSetDevice(1);
  cudaDeviceEnablePeerAccess(0, 0);

  // Run 2 warm-up kernels
  cudaSetDevice(1);
  warmUpKernel<<<1, 1>>>();
  cudaSetDevice(0);
  warmUpKernel<<<1, 1>>>();

  // Ensure all operations are completed
  cudaDeviceSynchronize();
}

void HybridEngine::Solve() {
  Precheck();
  if (!is_perfect_) {
    Init();
    Exec();
    Postproc();
  } else {
    std::cout << "Perfect Case: Skip all subsequent skeps. " << std::endl;
  }
}

auto HybridEngine::GetStableMatching() const -> std::vector<int> {
  std::vector<int> result = std::vector<int>(n_);
  for (int i = 0; i < n_; i++) {
    result[i] = stable_matching_[i];
  }
  return result;
  // return stable_matching_;
}

HybridEngine::~HybridEngine() {
  if (!is_perfect_) {
    CUDA_CHECK(cudaFree(device_pref_lists_w_));
    CUDA_CHECK(cudaFree(device_pref_lists_m_));
    CUDA_CHECK(cudaFree(device_rank_mtx_w_));

    CUDA_CHECK(cudaFree(device_next_proposed_w_));
    CUDA_CHECK(cudaFree(device_partner_rank_));

    CUDA_CHECK(cudaSetDevice(1));
    CUDA_CHECK(cudaDeviceDisablePeerAccess(0));
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceDisablePeerAccess(1));
  }
}

void HybridEngine::PrintMatching() const {
  std::cout << "Stable Matching: " << std::endl;
  for (int i = 0; i < n_; ++i) {
    std::cout << "( Man:" << i
              << " is paired with Woman:" << stable_matching_[i] << ") "
              << std::endl;
  }
  std::cout << std::endl;
}

} // namespace bamboosmp
