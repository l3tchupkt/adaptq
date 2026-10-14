#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>

static inline int select_top_mass_indices(const float *weights, int n,
                                          float mass_thresh, int *ord) {
  if (!weights || !ord || n <= 0)
    return 0;

  std::iota(ord, ord + n, 0);

  const int selected_count =
      std::min(n, std::max(1, (int)std::ceil(mass_thresh * (float)n)));

  if (selected_count < n) {
    std::nth_element(
        ord, ord + selected_count, ord + n,
        [&](int a, int b) { return weights[a] > weights[b]; });
    std::sort(ord, ord + selected_count,
              [&](int a, int b) { return weights[a] > weights[b]; });
  } else {
    std::sort(ord, ord + n,
              [&](int a, int b) { return weights[a] > weights[b]; });
  }

  return selected_count;
}
