#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <cfloat>

namespace adaptq {
namespace core {

/**
 * @brief Configuration parameters for online codebook calibration and optimization.
 */
struct CodebookCalibratorConfig {
    int bit_width{3};                       /// Quantization bit depth (typically 2, 3, or 4)
    size_t max_iterations{50};              /// Maximum Lloyd-Max iterative optimization passes
    double convergence_epsilon{1e-5};       /// Minimum change in distortion to continue iterating
    double ema_smoothing_factor{0.2};       /// Exponential moving average rate for online centroid updates
    size_t histogram_bins{512};             /// Resolution of streaming histogram for continuous calibration
    double initial_range_min{-3.5};         /// Initial lower bound for histogram tracking
    double initial_range_max{3.5};          /// Initial upper bound for histogram tracking
};

/**
 * @brief Empirical statistical moments of observed activation data.
 */
struct DistributionMoments {
    uint64_t sample_count{0};
    double mean{0.0};
    double variance{1.0};
    double std_dev{1.0};
    double skewness{0.0};
    double excess_kurtosis{0.0};
    double min_val{0.0};
    double max_val{0.0};
};

/**
 * @brief Results and metrics from a codebook calibration pass.
 */
struct CalibrationResult {
    bool converged{false};
    size_t iterations_run{0};
    double initial_distortion{0.0};
    double final_distortion{0.0};
    double distortion_reduction_pct{0.0};
    std::vector<float> centroids;
    std::vector<float> thresholds;
};

/**
 * @brief Online Codebook Calibrator & Lloyd-Max Adaptive Centroid Tuner.
 *
 * Provides online empirical distribution estimation, streaming moment calculation,
 * and adaptive Lloyd-Max optimal quantization codebook calibration for domain-skewed
 * transformer activations.
 */
class OnlineCodebookCalibrator {
public:
    explicit OnlineCodebookCalibrator(const CodebookCalibratorConfig& config = CodebookCalibratorConfig{})
        : config_(config), num_levels_(1 << config.bit_width) {
        if (config_.bit_width < 1 || config_.bit_width > 8) {
            throw std::invalid_argument("Bit width must be between 1 and 8");
        }
        if (config_.ema_smoothing_factor <= 0.0 || config_.ema_smoothing_factor > 1.0) {
            throw std::invalid_argument("EMA smoothing factor must be in (0.0, 1.0]");
        }

        initialize_default_centroids();
        reset_histogram();
    }

    /**
     * @brief Ingest a batch of continuous activation values into the streaming histogram.
     * @param data Pointer to continuous float activations.
     * @param count Number of elements in the buffer.
     */
    void ingest_samples(const float* data, size_t count) {
        if (!data || count == 0) return;

        for (size_t i = 0; i < count; ++i) {
            double x = static_cast<double>(data[i]);

            // Update running moments via Welford-style algorithm
            moments_.sample_count++;
            double delta = x - moments_.mean;
            moments_.mean += delta / moments_.sample_count;
            double delta2 = x - moments_.mean;
            M2_ += delta * delta2;

            double delta_cubed = delta * delta * delta;
            M3_ += delta_cubed * (moments_.sample_count - 1) * (moments_.sample_count - 2) / (moments_.sample_count * moments_.sample_count)
                   - 3.0 * delta * M2_ / moments_.sample_count;

            if (x < moments_.min_val || moments_.sample_count == 1) moments_.min_val = x;
            if (x > moments_.max_val || moments_.sample_count == 1) moments_.max_val = x;

            // Ingest into histogram
            if (x >= config_.initial_range_min && x <= config_.initial_range_max) {
                double normalized = (x - config_.initial_range_min) / (config_.initial_range_max - config_.initial_range_min);
                size_t bin = std::min(static_cast<size_t>(normalized * config_.histogram_bins), config_.histogram_bins - 1);
                histogram_[bin]++;
                total_histogram_samples_++;
            }
        }

        if (moments_.sample_count > 1) {
            moments_.variance = M2_ / (moments_.sample_count - 1);
            moments_.std_dev = std::sqrt(std::max(0.0, moments_.variance));
            if (moments_.std_dev > 1e-9) {
                moments_.skewness = (M3_ / moments_.sample_count) / std::pow(moments_.variance, 1.5);
            }
        }
    }

    /**
     * @brief Execute Lloyd-Max optimization using observed empirical distribution.
     * @param use_ema If true, blends newly optimized centroids with previous codebook via EMA.
     * @return CalibrationResult containing optimized centroids, thresholds, and MSE reduction.
     */
    CalibrationResult calibrate(bool use_ema = true) {
        if (total_histogram_samples_ == 0) {
            throw std::runtime_error("Cannot calibrate codebook with 0 ingested samples");
        }

        CalibrationResult result;
        result.initial_distortion = compute_distortion(centroids_);

        std::vector<float> current_centroids = centroids_;
        std::vector<float> current_thresholds(num_levels_ - 1, 0.0f);

        double prev_distortion = result.initial_distortion;
        bool has_converged = false;
        size_t iter = 0;

        for (iter = 0; iter < config_.max_iterations; ++iter) {
            // Step 1: Compute optimal nearest-neighbor decision thresholds (midpoints)
            for (size_t i = 0; i < num_levels_ - 1; ++i) {
                current_thresholds[i] = (current_centroids[i] + current_centroids[i + 1]) * 0.5f;
            }

            // Step 2: Compute optimal centroids (conditional expectation within each partition)
            std::vector<double> sum_x(num_levels_, 0.0);
            std::vector<double> count_x(num_levels_, 0.0);

            double bin_width = (config_.initial_range_max - config_.initial_range_min) / config_.histogram_bins;

            for (size_t b = 0; b < config_.histogram_bins; ++b) {
                uint64_t freq = histogram_[b];
                if (freq == 0) continue;

                double bin_center = config_.initial_range_min + (b + 0.5) * bin_width;

                // Find partition index for bin_center
                size_t part = 0;
                while (part < num_levels_ - 1 && bin_center >= current_thresholds[part]) {
                    part++;
                }

                sum_x[part] += bin_center * freq;
                count_x[part] += freq;
            }

            for (size_t i = 0; i < num_levels_; ++i) {
                if (count_x[i] > 0.0) {
                    current_centroids[i] = static_cast<float>(sum_x[i] / count_x[i]);
                }
            }

            // Ensure monotonicity
            for (size_t i = 1; i < num_levels_; ++i) {
                if (current_centroids[i] <= current_centroids[i - 1]) {
                    current_centroids[i] = current_centroids[i - 1] + 1e-4f;
                }
            }

            double curr_distortion = compute_distortion(current_centroids);
            double dist_diff = std::abs(prev_distortion - curr_distortion);

            if (dist_diff < config_.convergence_epsilon) {
                has_converged = true;
                break;
            }
            prev_distortion = curr_distortion;
        }

        // Apply EMA smoothing to prevent abrupt codebook jumps if requested
        if (use_ema) {
            for (size_t i = 0; i < num_levels_; ++i) {
                centroids_[i] = static_cast<float>(
                    (1.0 - config_.ema_smoothing_factor) * centroids_[i] +
                    config_.ema_smoothing_factor * current_centroids[i]
                );
            }
        } else {
            centroids_ = current_centroids;
        }

        // Recompute thresholds from finalized centroids
        thresholds_.resize(num_levels_ - 1);
        for (size_t i = 0; i < num_levels_ - 1; ++i) {
            thresholds_[i] = (centroids_[i] + centroids_[i + 1]) * 0.5f;
        }

        result.converged = has_converged;
        result.iterations_run = iter + 1;
        result.final_distortion = compute_distortion(centroids_);
        result.centroids = centroids_;
        result.thresholds = thresholds_;

        if (result.initial_distortion > 1e-9) {
            result.distortion_reduction_pct = 100.0 * (result.initial_distortion - result.final_distortion) / result.initial_distortion;
        } else {
            result.distortion_reduction_pct = 0.0;
        }

        return result;
    }

    /**
     * @brief Quantize a continuous activation scalar to its discrete codebook bin index.
     * @param value Continuous float activation.
     * @return Discrete bin index in [0, num_levels_ - 1].
     */
    uint8_t quantize(float value) const {
        for (size_t i = 0; i < thresholds_.size(); ++i) {
            if (value < thresholds_[i]) {
                return static_cast<uint8_t>(i);
            }
        }
        return static_cast<uint8_t>(num_levels_ - 1);
    }

    /**
     * @brief Dequantize a discrete codebook index back to its reconstructed centroid value.
     * @param code Discrete bin index.
     * @return Reconstructed centroid float.
     */
    float dequantize(uint8_t code) const {
        if (code >= num_levels_) {
            return centroids_.back();
        }
        return centroids_[code];
    }

    /**
     * @brief Get currently calibrated codebook centroids.
     */
    const std::vector<float>& centroids() const { return centroids_; }

    /**
     * @brief Get currently calibrated decision thresholds.
     */
    const std::vector<float>& thresholds() const { return thresholds_; }

    /**
     * @brief Get streaming distribution moments.
     */
    const DistributionMoments& moments() const { return moments_; }

    /**
     * @brief Reset histogram and moment counters.
     */
    void reset_histogram() {
        histogram_.assign(config_.histogram_bins, 0);
        total_histogram_samples_ = 0;
        moments_ = DistributionMoments{};
        M2_ = 0.0;
        M3_ = 0.0;
    }

private:
    void initialize_default_centroids() {
        centroids_.resize(num_levels_);
        // Linear uniform initialization across [-2.0, 2.0]
        float step = 4.0f / static_cast<float>(num_levels_);
        float start = -2.0f + step * 0.5f;
        for (size_t i = 0; i < num_levels_; ++i) {
            centroids_[i] = start + static_cast<float>(i) * step;
        }

        thresholds_.resize(num_levels_ - 1);
        for (size_t i = 0; i < num_levels_ - 1; ++i) {
            thresholds_[i] = (centroids_[i] + centroids_[i + 1]) * 0.5f;
        }
    }

    double compute_distortion(const std::vector<float>& centroids) const {
        if (total_histogram_samples_ == 0) return 0.0;

        double total_err = 0.0;
        double bin_width = (config_.initial_range_max - config_.initial_range_min) / config_.histogram_bins;

        for (size_t b = 0; b < config_.histogram_bins; ++b) {
            uint64_t freq = histogram_[b];
            if (freq == 0) continue;

            double bin_center = config_.initial_range_min + (b + 0.5) * bin_width;

            // Find closest centroid
            double min_dist_sq = DBL_MAX;
            for (float c : centroids) {
                double diff = bin_center - static_cast<double>(c);
                double d_sq = diff * diff;
                if (d_sq < min_dist_sq) {
                    min_dist_sq = d_sq;
                }
            }
            total_err += min_dist_sq * freq;
        }

        return total_err / total_histogram_samples_;
    }

    CodebookCalibratorConfig config_;
    size_t num_levels_;
    std::vector<float> centroids_;
    std::vector<float> thresholds_;
    std::vector<uint64_t> histogram_;
    uint64_t total_histogram_samples_{0};
    DistributionMoments moments_;
    double M2_{0.0};
    double M3_{0.0};
};

} // namespace core
} // namespace adaptq
