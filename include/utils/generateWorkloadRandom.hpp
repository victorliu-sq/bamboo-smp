#include "utils.hpp"
#include <algorithm>
#include <cstdlib>
#include <random>
#include <thread>
#include <utils/utils.hpp>
#include <vector>

// -------------------- Random Case ----------------------------
// ----------------------------------------------------------------
static void GeneratePrefListsRandomOneRow(const int n, const int group_size,
                                          std::vector<int> &pl,
                                          std::vector<int> &ids) {
  int num_groups = (n + group_size - 1) / group_size;
  int cur_group_size;
  for (int group_id = 0; group_id < num_groups; group_id++) {
    if (group_id == num_groups - 1 && n % group_size != 0) {
      cur_group_size = n % group_size;
    } else {
      cur_group_size = group_size;
    }
    std::vector<int> cur_ids(cur_group_size, 0);
    for (int i = 0; i < cur_ids.size(); i++) {
      cur_ids[i] = ids[group_id * group_size + i];
    }

    std::mt19937 rng(static_cast<unsigned int>(getNanoSecond()));
    std::shuffle(cur_ids.begin(), cur_ids.end(), rng);

    for (int i = 0; i < cur_ids.size(); i++) {
      pl[group_id * group_size + i] = cur_ids[i];
    }
  }
}

static PreferenceLists GeneratePrefListsRandom(const int n,
                                               const int group_size) {
  PreferenceLists pls(n, std::vector<int>(n));
  std::vector<int> ids(n);
  for (int i = 0; i < n; i++) {
    ids[i] = i;
  }
  std::srand(static_cast<unsigned int>(getNanoSecond()));
  std::random_shuffle(ids.begin(), ids.end());
  int maxThreads =
      std::thread::hardware_concurrency(); // Use hardware concurrency as a
                                           // limit.
  std::vector<std::thread> threads;
  threads.reserve(maxThreads);

  for (int i = 0; i < n; ++i) {
    if (threads.size() >= maxThreads) {
      // Wait for all the threads in this batch to complete.
      for (auto &th : threads) {
        th.join();
      }
      threads.clear();
    }
    threads.emplace_back(GeneratePrefListsRandomOneRow, n, group_size,
                         std::ref(pls[i]), std::ref(ids));
  }

  // Ensure any remaining threads are joined.
  for (auto &th : threads) {
    th.join();
  }

  return pls;
}
