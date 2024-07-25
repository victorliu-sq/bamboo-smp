#include <utils/utils.hpp>
#include <vector>

// --------------------------- Perfect Case -------------------------------
// ------------------------------------------------------------------------
// Functions to return a serial workload (both Preference Lists for men and
// women) Function to generate a preference list for a given 'm' and 'n'

static void GeneratePerfListPerfectRow(int m, int n,
                                       std::vector<int> first_choices,
                                       PreferenceLists &pls) {
  int first_choice = first_choices[m];
  for (int i = 0; i < n; i++) {
    pls[m][i] = (first_choice + i) % n;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(pls[m].begin() + 1, pls[m].end(), g);
}
// Function to manage threads and return the preference list matrix
static PreferenceLists GeneratePrefListsPerfect(int n) {
  PreferenceLists pls(n, std::vector<int>(n));

  std::vector<std::thread> threads;
  std::vector<int> first_choices(n, 0);

  for (int m = 0; m < n; m++) {
    first_choices[m] = m;
  }

  std::random_device rd; // Seed
  std::mt19937 g(rd());  // Mersenne Twister engine

  // Shuffle the vector
  std::shuffle(first_choices.begin(), first_choices.end(), g);
  // Launch threads
  for (int m = 0; m < n; m++) {
    threads.emplace_back(GeneratePerfListPerfectRow, m, n, first_choices,
                         std::ref(pls));
  }

  // Join threads
  for (auto &th : threads) {
    th.join();
  }

  return pls;
}