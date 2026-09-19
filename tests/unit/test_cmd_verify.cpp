#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

#include "../../cli/cmd_verify.h"

using namespace adaptq;

TEST_CASE("TensorVerificationEngine: perfect match validation", "[cli][verify]") {
    std::vector<float> vec = {1.0f, -2.0f, 3.5f, -0.5f, 4.0f};
    VerificationResult res = TensorVerificationEngine::verify(vec.data(), vec.data(), vec.size());

    REQUIRE(res.passed == true);
    REQUIRE(res.elements_evaluated == 5);
    REQUIRE(res.mse == 0.0);
    REQUIRE(res.max_abs_error == 0.0);
    REQUIRE_THAT(res.cosine_similarity, Catch::Matchers::WithinRel(1.0, 1e-6));
    REQUIRE(res.snr_db > 100.0); // Inf/perfect SNR sentinel
}

TEST_CASE("TensorVerificationEngine: acceptable perturbation passes", "[cli][verify]") {
    std::vector<float> base = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> cand = {1.0005f, 1.9995f, 3.0002f, 3.9998f};

    VerificationResult res = TensorVerificationEngine::verify(base.data(), cand.data(), base.size());
    REQUIRE(res.passed == true);
    REQUIRE(res.mse < 1e-3);
    REQUIRE(res.cosine_similarity > 0.999);
    REQUIRE(res.snr_db > 30.0);
}

TEST_CASE("TensorVerificationEngine: threshold violation detection", "[cli][verify]") {
    std::vector<float> base = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> corrupted = {2.5f, -1.0f, 0.0f, 8.0f};

    VerificationThresholds thresholds;
    thresholds.max_mse = 0.01;
    thresholds.min_cosine_similarity = 0.95;

    VerificationResult res = TensorVerificationEngine::verify(
        base.data(), corrupted.data(), base.size(), thresholds
    );

    REQUIRE(res.passed == false);
    REQUIRE(!res.failure_reason.empty());
}

TEST_CASE("TensorVerificationEngine: null pointer and empty handling", "[cli][verify]") {
    std::vector<float> vec = {1.0f, 2.0f};
    VerificationResult r1 = TensorVerificationEngine::verify(nullptr, vec.data(), 2);
    REQUIRE(r1.passed == false);

    VerificationResult r2 = TensorVerificationEngine::verify(vec.data(), nullptr, 2);
    REQUIRE(r2.passed == false);

    VerificationResult r3 = TensorVerificationEngine::verify(vec.data(), vec.data(), 0);
    REQUIRE(r3.passed == false);
}

TEST_CASE("TensorVerificationEngine: JSON report serialization", "[cli][verify]") {
    std::vector<float> base = {1.0f, 2.0f, 3.0f};
    std::vector<float> cand = {1.01f, 2.01f, 3.01f};

    VerificationResult res = TensorVerificationEngine::verify(base.data(), cand.data(), base.size());
    std::string json_str = TensorVerificationEngine::to_json(res);

    REQUIRE(json_str.find("\"passed\": true") != std::string::npos);
    REQUIRE(json_str.find("\"elements_evaluated\": 3") != std::string::npos);
    REQUIRE(json_str.find("\"mse\":") != std::string::npos);
    REQUIRE(json_str.find("\"cosine_similarity\":") != std::string::npos);
}
