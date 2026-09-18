#include <catch2/catch_test_macros.hpp>
#include "../../core/codebook_calibrator.h"
#include <vector>
#include <random>
#include <cmath>

using namespace adaptq::core;

TEST_CASE("OnlineCodebookCalibrator: initialization and config validation", "[core][codebook]") {
    SECTION("Valid config") {
        CodebookCalibratorConfig cfg;
        cfg.bit_width = 3;
        cfg.ema_smoothing_factor = 0.5;
        REQUIRE_NOTHROW(OnlineCodebookCalibrator(cfg));
    }

    SECTION("Invalid bit widths") {
        CodebookCalibratorConfig cfg;
        cfg.bit_width = 0;
        REQUIRE_THROWS_AS(OnlineCodebookCalibrator(cfg), std::invalid_argument);

        cfg.bit_width = 9;
        REQUIRE_THROWS_AS(OnlineCodebookCalibrator(cfg), std::invalid_argument);
    }

    SECTION("Invalid EMA smoothing factor") {
        CodebookCalibratorConfig cfg;
        cfg.ema_smoothing_factor = 0.0;
        REQUIRE_THROWS_AS(OnlineCodebookCalibrator(cfg), std::invalid_argument);

        cfg.ema_smoothing_factor = 1.5;
        REQUIRE_THROWS_AS(OnlineCodebookCalibrator(cfg), std::invalid_argument);
    }
}

TEST_CASE("OnlineCodebookCalibrator: streaming moments calculation", "[core][codebook]") {
    CodebookCalibratorConfig cfg;
    cfg.bit_width = 3;
    OnlineCodebookCalibrator calibrator(cfg);

    // Generate normal distribution data with mean 0.5 and stddev 1.2
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.5f, 1.2f);

    std::vector<float> samples(2000);
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = dist(rng);
    }

    calibrator.ingest_samples(samples.data(), samples.size());

    const auto& m = calibrator.moments();
    REQUIRE(m.sample_count == 2000);
    REQUIRE(std::abs(m.mean - 0.5) < 0.1);
    REQUIRE(std::abs(m.std_dev - 1.2) < 0.1);
    REQUIRE(m.min_val < m.mean);
    REQUIRE(m.max_val > m.mean);
}

TEST_CASE("OnlineCodebookCalibrator: Lloyd-Max convergence on Gaussian distribution", "[core][codebook]") {
    CodebookCalibratorConfig cfg;
    cfg.bit_width = 3; // 8 centroids
    cfg.max_iterations = 40;
    cfg.convergence_epsilon = 1e-4;
    cfg.ema_smoothing_factor = 1.0; // full update without historical blend
    OnlineCodebookCalibrator calibrator(cfg);

    std::mt19937 rng(1337);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> samples(5000);
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = dist(rng);
    }

    calibrator.ingest_samples(samples.data(), samples.size());

    CalibrationResult res = calibrator.calibrate(false);
    REQUIRE(res.iterations_run > 0);
    REQUIRE(res.final_distortion < res.initial_distortion);
    REQUIRE(res.distortion_reduction_pct > 0.0);
    REQUIRE(res.centroids.size() == 8);
    REQUIRE(res.thresholds.size() == 7);

    // Centroids must be monotonically increasing
    for (size_t i = 1; i < res.centroids.size(); ++i) {
        REQUIRE(res.centroids[i] > res.centroids[i - 1]);
    }

    // Centroids should be roughly symmetric around 0 for standard normal
    for (size_t i = 0; i < 4; ++i) {
        float left = res.centroids[i];
        float right = res.centroids[7 - i];
        REQUIRE(std::abs(left + right) < 0.35f);
    }
}

TEST_CASE("OnlineCodebookCalibrator: quantization and dequantization fidelity", "[core][codebook]") {
    CodebookCalibratorConfig cfg;
    cfg.bit_width = 2; // 4 centroids
    cfg.ema_smoothing_factor = 1.0;
    OnlineCodebookCalibrator calibrator(cfg);

    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> samples(2000);
    for (size_t i = 0; i < samples.size(); ++i) {
        samples[i] = dist(rng);
    }

    calibrator.ingest_samples(samples.data(), samples.size());
    calibrator.calibrate(false);

    // Test quantize and dequantize
    uint8_t q_neg = calibrator.quantize(-3.0f);
    REQUIRE(q_neg == 0);
    REQUIRE(calibrator.dequantize(q_neg) == calibrator.centroids()[0]);

    uint8_t q_pos = calibrator.quantize(3.0f);
    REQUIRE(q_pos == 3);
    REQUIRE(calibrator.dequantize(q_pos) == calibrator.centroids()[3]);

    uint8_t q_zero = calibrator.quantize(0.0f);
    REQUIRE((q_zero == 1 || q_zero == 2));
}

TEST_CASE("OnlineCodebookCalibrator: EMA adaptation and histogram reset", "[core][codebook]") {
    CodebookCalibratorConfig cfg;
    cfg.bit_width = 2;
    cfg.ema_smoothing_factor = 0.3;
    OnlineCodebookCalibrator calibrator(cfg);

    std::vector<float> batch1(500, 1.0f);
    calibrator.ingest_samples(batch1.data(), batch1.size());

    CalibrationResult res1 = calibrator.calibrate(true);
    REQUIRE(res1.centroids.size() == 4);

    calibrator.reset_histogram();
    REQUIRE(calibrator.moments().sample_count == 0);

    // Calibrating empty histogram throws
    REQUIRE_THROWS_AS(calibrator.calibrate(), std::runtime_error);
}
