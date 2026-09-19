#pragma once

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace adaptq {

struct VerificationThresholds {
    double max_mse = 1e-3;
    double min_cosine_similarity = 0.99;
    double min_snr_db = 25.0;
    double max_abs_error = 0.05;
};

struct VerificationResult {
    bool passed = false;
    double mse = 0.0;
    double cosine_similarity = 0.0;
    double snr_db = 0.0;
    double max_abs_error = 0.0;
    size_t elements_evaluated = 0;
    std::string failure_reason;
};

class TensorVerificationEngine {
public:
    static VerificationResult verify(
        const float *baseline,
        const float *candidate,
        size_t count,
        const VerificationThresholds &thresholds = VerificationThresholds{}
    ) {
        VerificationResult res;
        res.elements_evaluated = count;

        if (!baseline || !candidate || count == 0) {
            res.passed = false;
            res.failure_reason = "Null buffer or zero element count";
            return res;
        }

        double sum_sq_diff = 0.0;
        double sum_sq_base = 0.0;
        double sum_sq_cand = 0.0;
        double dot_prod = 0.0;
        double max_err = 0.0;

        for (size_t i = 0; i < count; ++i) {
            double b = static_cast<double>(baseline[i]);
            double c = static_cast<double>(candidate[i]);
            double diff = std::abs(b - c);

            if (diff > max_err) {
                max_err = diff;
            }

            sum_sq_diff += diff * diff;
            sum_sq_base += b * b;
            sum_sq_cand += c * c;
            dot_prod += b * c;
        }

        res.mse = sum_sq_diff / static_cast<double>(count);
        res.max_abs_error = max_err;

        // Cosine similarity
        double denom = std::sqrt(sum_sq_base) * std::sqrt(sum_sq_cand);
        if (denom > 1e-12) {
            res.cosine_similarity = dot_prod / denom;
        } else {
            res.cosine_similarity = (sum_sq_diff < 1e-12) ? 1.0 : 0.0;
        }

        // SNR in dB
        if (sum_sq_diff < 1e-15) {
            res.snr_db = 999.0; // Perfect match
        } else if (sum_sq_base < 1e-15) {
            res.snr_db = 0.0;
        } else {
            res.snr_db = 10.0 * std::log10(sum_sq_base / sum_sq_diff);
        }

        // Check against thresholds
        if (res.mse > thresholds.max_mse) {
            res.passed = false;
            res.failure_reason = "MSE exceeded maximum threshold";
            return res;
        }
        if (res.cosine_similarity < thresholds.min_cosine_similarity) {
            res.passed = false;
            res.failure_reason = "Cosine similarity fell below minimum threshold";
            return res;
        }
        if (res.snr_db < thresholds.min_snr_db) {
            res.passed = false;
            res.failure_reason = "Signal-to-noise ratio fell below minimum threshold";
            return res;
        }
        if (res.max_abs_error > thresholds.max_abs_error) {
            res.passed = false;
            res.failure_reason = "Max absolute error exceeded threshold";
            return res;
        }

        res.passed = true;
        return res;
    }

    static std::string to_json(const VerificationResult &res) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(6);
        ss << "{\n";
        ss << "  \"passed\": " << (res.passed ? "true" : "false") << ",\n";
        ss << "  \"elements_evaluated\": " << res.elements_evaluated << ",\n";
        ss << "  \"mse\": " << res.mse << ",\n";
        ss << "  \"cosine_similarity\": " << res.cosine_similarity << ",\n";
        ss << "  \"snr_db\": " << res.snr_db << ",\n";
        ss << "  \"max_abs_error\": " << res.max_abs_error << ",\n";
        ss << "  \"failure_reason\": \"" << res.failure_reason << "\"\n";
        ss << "}";
        return ss.str();
    }
};

} // namespace adaptq
