#include <iostream>
#include <smp/BambooSMP.cuh>
#include <utils/utils.hpp>

// ********************* Cohabitation Hybrid *************************
// *******************************************************************
namespace bamboosmp {

void HybridEngine::Postproc() {
  std::cout << "Postprocessing ... " << std::endl;
  int *device_match;
  int *device_partner_rank;

  if (atomic_host_terminate_flag_.load() == flag_gpu) {
    CUDA_CHECK(cudaSetDevice(0));

    CUDA_CHECK(cudaMemcpy(host_partner_rank_, device_partner_rank_,
                          n_ * sizeof(int), cudaMemcpyDeviceToHost));
  } else {
    std::cout << "Use PartnerRank on CPU" << std::endl;
  }
  for (int w = 0; w < n_; w++) {
    int m = smp_->flatten_pref_lists_w_[w * n_ + host_partner_rank_[w]];
    stable_matching_[m] = w;
  }

  cudaFree(device_match);
}

} // namespace bamboosmp
