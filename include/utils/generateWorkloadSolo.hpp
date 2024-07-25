#include <algorithm>
#include <random>
#include <utils/utils.hpp>
#include <vector>

// --------------------------- Solo Case --------------------------------
// ------------------------------------------------------------------------
// Functions to return a serial workload (both Preference Lists for men and
// women) Function to generate a preference list for a given 'm' and 'n'

static void GeneratePrefListsManSoloRow(int m, int n, PreferenceLists &plM) {
  std::vector<int> preference_list(n);
  int original_m = m;
  if (original_m == n - 1) {
    m = n - 2;
  }
  int w1 = m, w2 = (m - 1 + n - 1) % (n - 1), w_last = n - 1;
  preference_list[0] = w1;
  preference_list[n - 2] = w2;
  preference_list[n - 1] = w_last;
  int w = 0, rank = 1;
  while (rank < n - 2) {
    while (w == w1 || w == w2) {
      w++;
    }
    preference_list[rank] = w;
    w++;
    rank++;
  }
  // randomly shuffle all elements from index:1 to index:n-3
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(preference_list.begin() + 1, preference_list.begin() + n - 2, g);

  plM[original_m] = preference_list;
}

// Function to manage threads and return the preference list matrix
static PreferenceLists GeneratePrefListsManSolo(int n) {
  PreferenceLists plM(n);
  std::vector<std::thread> threads;

  // Launch threads
  for (int m = 0; m < n; m++) {
    threads.emplace_back(GeneratePrefListsManSoloRow, m, n, std::ref(plM));
  }

  // Join threads
  for (auto &th : threads) {
    th.join();
  }

  return plM;
}

static void GeneratePrefListsWomanSoloRow(int m, int n,
                                          std::vector<std::vector<int>> &plW) {
  std::vector<int> preference_list(n);
  if (m < n - 1) {
    int w1 = (m + 1) % (n - 1), w2 = (w1 + n - 1) % n;
    preference_list[0] = w1;
    preference_list[1] = w2;
    int w = 0, rank = 2;
    while (rank < n) {
      while (w == w1 || w == w2) {
        w++;
      }
      preference_list[rank] = w;
      w++;
      rank++;
    }
  } else {
    for (int rank = 0; rank < n - 1; rank++) {
      preference_list[rank] = n - 1 - rank;
    }
  }

  // randomly shuffle all elements from index:1 to index:n-3
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(preference_list.begin() + 2, preference_list.end(), g);
  plW[m] = preference_list;
}

static PreferenceLists GeneratePrefListsWomanSolo(int n) {
  PreferenceLists plW(n);
  std::vector<std::thread> threads;

  // Launch threads
  for (int m = 0; m < n; m++) {
    threads.emplace_back(GeneratePrefListsWomanSoloRow, m, n, std::ref(plW));
  }

  // Join threads
  for (auto &th : threads) {
    th.join();
  }

  return plW;
}

// --------------------------- Solo Case --------------------------------
// ------------------------------------------------------------------------
// Functions to return a serial workload (both Preference Lists for men and
// women) Function to generate a preference list for a given 'm' and 'n'

static void GeneratePrefListsManSoloRandRow(int m, int n,
                                            std::vector<std::vector<int>> &plM,
                                            std::vector<int> men_map,
                                            std::vector<int> women_map) {
  std::vector<int> preference_list(n);
  int original_m = m;

  if (original_m == n - 1) {
    m = n - 2;
  }
  int w1 = m, w2 = (m - 1 + n - 1) % (n - 1), w_last = n - 1;
  // preference_list[0] = w1;
  // preference_list[n - 2] = w2;
  // preference_list[n - 1] = w_last;
  preference_list[0] = women_map[w1];
  preference_list[n - 2] = women_map[w2];
  preference_list[n - 1] = women_map[w_last];
  int w = 0, rank = 1;
  while (rank < n - 2) {
    while (w == w1 || w == w2) {
      w++;
    }
    // randomized label of woman
    // preference_list[rank] = w;
    preference_list[rank] = women_map[w];
    w++;
    rank++;
  }
  // randomly shuffle all elements from index:1 to index:n-3
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(preference_list.begin() + 1, preference_list.begin() + n - 2, g);

  // randomized label of man
  int mapped_m = men_map[original_m];
  // plM[original_m] = preference_list;
  plM[mapped_m] = preference_list;
}

static void GeneratePrefListsWomanSoloRandRow(
    int woman, int n, std::vector<std::vector<int>> &plW,
    std::vector<int> men_map, std::vector<int> women_map) {
  // std::cout << "Init PrefLists for woman " << woman << std::endl;
  std::vector<int> preference_list(n);
  if (woman < n - 1) {
    int m1 = (woman + 1) % (n - 1), m2 = (m1 + n - 1) % n;
    preference_list[0] = men_map[m1];
    preference_list[1] = men_map[m2];
    int m = 0, rank = 2;
    while (rank < n) {
      while (m == m1 || m == m2) {
        m++;
      }
      preference_list[rank] = men_map[m];
      m++;
      rank++;
    }
  } else {
    for (int rank = 0; rank < n - 1; rank++) {
      preference_list[rank] = rank;
    }
  }

  // randomly shuffle all elements from index:1 to index:n-3
  std::random_device rd;
  std::mt19937 g(rd());
  if (woman < n - 1) {
    std::shuffle(preference_list.begin() + 2, preference_list.end(), g);
  } else {
    std::shuffle(preference_list.begin(), preference_list.end(), g);
  }
  // plW[woman] = preference_list;
  plW[women_map[woman]] = preference_list;
}

static void GenerateRandomizedLabelPreflistsSolo(int n, PreferenceLists &plM,
                                                 PreferenceLists &plW) {

  plM.resize(n);
  plW.resize(n);
  std::vector<int> men_map(n, 0);
  std::vector<int> women_map(n, 0);

  std::vector<std::thread> threadsM;
  std::vector<std::thread> threadsW;

  // Fill the vector with numbers from 0 to n-1
  for (int i = 0; i < n; ++i) {
    women_map[i] = i;
    men_map[i] = i;
  }

  // Obtain a random number generator
  std::random_device rd; // Seed
  std::mt19937 g(rd());  // Mersenne Twister engine

  // Shuffle the vector
  std::shuffle(men_map.begin(), men_map.end(), g);
  std::shuffle(women_map.begin(), women_map.end(), g);

  // Print the shuffled vector
  // std::cout << "Shuffled Men: ";
  // for (const auto &man : men_map) {
  //   std::cout << man << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "Shuffled Women: ";
  // for (const auto &woman : women_map) {
  //   std::cout << woman << " ";
  // }

  // std::cout << std::endl;

  // std::cout << "Generate PLM: ";
  // Launch threads for Men preferences
  for (int m = 0; m < n; m++) {
    threadsM.emplace_back(GeneratePrefListsManSoloRandRow, m, n, std::ref(plM),
                          men_map, women_map);
  }

  // Join threads
  for (auto &th : threadsM) {
    th.join();
  }

  // Clear the threads vector before launching new threads
  // Otherwise, you get segfaults
  // threads.clear();

  // std::cout << "Generate PLW: ";
  // Launch threads for Women preferences
  for (int w = 0; w < n; w++) {
    threadsW.emplace_back(GeneratePrefListsWomanSoloRandRow, w, n,
                          std::ref(plW), men_map, women_map);
  }

  // Join threads
  for (auto &th : threadsW) {
    th.join();
  }
}
