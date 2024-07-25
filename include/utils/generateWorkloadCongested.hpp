#include "utils.hpp"
#include <algorithm>
#include <cstdlib>
#include <random>
#include <thread>
#include <utils/utils.hpp>
#include <vector>

// -------------------- Congested Case ----------------------------
// ----------------------------------------------------------------
static PreferenceLists GeneratePrefListsCongested(int n) {
  PreferenceLists pls(n, std::vector<int>(n, 0));

  // Using random_shuffle for C++98 compatibility
  std::srand(static_cast<unsigned int>(getNanoSecond()));

  // (2) For each row: if i != j, randomly assign all numbers from 0 to n - 1 to
  // its left values
  std::vector<int> row_values(n);
  for (int i = 0; i < n; i++) {
    // printf("Current Row: %d\n", i);
    row_values[i] = i;
  }

  std::random_shuffle(row_values.begin(), row_values.end());

  for (int i = 0; i < n; ++i) {
    pls[i] = row_values;
  }

  return pls;
}
