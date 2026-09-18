#include <catch2/catch_test_macros.hpp>
#include "../../attention/sparse_selection.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

static float selected_mass(const std::vector<float> &weights,
                           const std::vector<int> &ord, int count) {
    float mass = 0.f;
    for (int i = 0; i < count; ++i)
        mass += weights[ord[i]];
    return mass;
}

TEST_CASE("Sparse V partial selection matches full-sort ordering",
          "[attention][sparse][selection]") {
    const std::vector<int> sizes = {1, 2, 3, 7, 8, 9, 16, 31, 32, 47, 48, 63, 64};
    const std::vector<float> thresholds = {0.01f, 0.10f, 0.50f, 0.75f, 0.95f, 1.0f};

    for (int n : sizes) {
        std::vector<float> weights(n);
        float total = 0.f;
        for (int i = 0; i < n; ++i) {
            weights[i] = 0.25f
                       + 0.013f * (float)(i + 1)
                       + 0.007f * (float)((i * 17) % 23);
            total += weights[i];
        }
        for (float &w : weights)
            w /= total;

        std::vector<int> full(n);
        std::iota(full.begin(), full.end(), 0);
        std::sort(full.begin(), full.end(),
                  [&](int a, int b) { return weights[a] > weights[b]; });

        for (float threshold : thresholds) {
            std::vector<int> selected(n, -1);
            const int count =
                select_top_mass_indices(weights.data(), n, threshold, selected.data());

            const int expected_count =
                std::min(n, std::max(1, (int)std::ceil(threshold * (float)n)));
            REQUIRE(count == expected_count);

            for (int i = 0; i < count; ++i)
                REQUIRE(selected[i] == full[i]);

            const float mass = selected_mass(weights, selected, count);
            REQUIRE(mass + 1e-6f >= std::min(1.0f, threshold));
        }
    }
}

TEST_CASE("Sparse V selector handles invalid and degenerate inputs",
          "[attention][sparse][selection][stability]") {
    int ord[4] = {-1, -1, -1, -1};
    float weights[4] = {0.1f, 0.2f, 0.3f, 0.4f};

    REQUIRE(select_top_mass_indices(nullptr, 4, 0.75f, ord) == 0);
    REQUIRE(select_top_mass_indices(weights, 0, 0.75f, ord) == 0);
    REQUIRE(select_top_mass_indices(weights, 4, 0.75f, nullptr) == 0);

    REQUIRE(select_top_mass_indices(weights, 4, 0.0f, ord) == 1);
    REQUIRE(ord[0] == 3);

    REQUIRE(select_top_mass_indices(weights, 4, 1.0f, ord) == 4);
    REQUIRE(ord[0] == 3);
    REQUIRE(ord[1] == 2);
    REQUIRE(ord[2] == 1);
    REQUIRE(ord[3] == 0);
}
