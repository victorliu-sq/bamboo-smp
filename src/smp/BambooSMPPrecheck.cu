#include <iostream>
#include <set>
#include <smp/BambooSMP.cuh>
#include <utils/utils.hpp>

// ********************* Cohabitation Hybrid *************************
// *******************************************************************
namespace bamboosmp {

void HybridEngine::Precheck() {
  // stable_matching_.resize(n_);
  CUDA_CHECK(cudaHostAlloc(&stable_matching_, n_ * sizeof(int), 0));
  std::set<int> topChoicesSet;

  is_perfect_ = true;
  for (int m = 0; m < n_; ++m) {
    int topChoice =
        smp_->flatten_pref_lists_m_[m * n_]; // Assuming 0-based index
    topChoicesSet.insert(topChoice);
    stable_matching_[m] = topChoice;
  }

  if (topChoicesSet.size() != n_) {
    is_perfect_ = false;
  }
}

} // namespace bamboosmp
