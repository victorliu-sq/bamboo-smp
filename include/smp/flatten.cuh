#pragma once
#include <utils/utils.hpp>

void FlattenRowHost(const PreferenceLists &pls, int *flat_pl, int row, int n);

int *ParallelFlattenHost(const PreferenceLists &pls, int n);