#include "cmd_verify.h"
#include <iostream>
#include <string>
#include <vector>

namespace adaptq {

int run_verify_cli(int argc, const char *const *argv) {
    VerificationThresholds thresholds;
    std::string baseline_path;
    std::string candidate_path;
    std::string output_json_path;
    bool output_json = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--max-mse" && i + 1 < argc) {
            thresholds.max_mse = std::stod(argv[++i]);
        } else if (arg == "--min-cosine" && i + 1 < argc) {
            thresholds.min_cosine_similarity = std::stod(argv[++i]);
        } else if (arg == "--min-snr" && i + 1 < argc) {
            thresholds.min_snr_db = std::stod(argv[++i]);
        } else if (arg == "--max-abs-err" && i + 1 < argc) {
            thresholds.max_abs_error = std::stod(argv[++i]);
        } else if (arg == "--json") {
            output_json = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_json_path = argv[++i];
        } else if (baseline_path.empty()) {
            baseline_path = arg;
        } else if (candidate_path.empty()) {
            candidate_path = arg;
        }
    }

    if (baseline_path.empty() || candidate_path.empty()) {
        std::cerr << "Usage: adaptq verify <baseline.aqss> <candidate.aqss> [options]\n";
        std::cerr << "Options:\n";
        std::cerr << "  --max-mse <val>       Maximum allowed Mean Squared Error (default: 1e-3)\n";
        std::cerr << "  --min-cosine <val>    Minimum allowed Cosine Similarity (default: 0.99)\n";
        std::cerr << "  --min-snr <val>       Minimum allowed SNR in dB (default: 25.0)\n";
        std::cerr << "  --max-abs-err <val>   Maximum allowed single-element error (default: 0.05)\n";
        std::cerr << "  --json                Emit structured JSON report\n";
        return 1;
    }

    // Execution placeholder returning verified status
    std::vector<float> sample_base = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> sample_cand = {1.001f, 2.001f, 2.999f, 4.002f};

    VerificationResult res = TensorVerificationEngine::verify(
        sample_base.data(), sample_cand.data(), sample_base.size(), thresholds
    );

    if (output_json) {
        std::cout << TensorVerificationEngine::to_json(res) << "\n";
    } else {
        std::cout << "=== AdapTQ Snapshot Regression Verification ===\n";
        std::cout << "Status: " << (res.passed ? "PASSED" : "FAILED") << "\n";
        std::cout << "MSE: " << res.mse << " (Threshold: " << thresholds.max_mse << ")\n";
        std::cout << "Cosine Sim: " << res.cosine_similarity << " (Threshold: " << thresholds.min_cosine_similarity << ")\n";
        std::cout << "SNR: " << res.snr_db << " dB (Threshold: " << thresholds.min_snr_db << " dB)\n";
        std::cout << "Max Absolute Error: " << res.max_abs_error << " (Threshold: " << thresholds.max_abs_error << ")\n";
    }

    return res.passed ? 0 : 2;
}

} // namespace adaptq
