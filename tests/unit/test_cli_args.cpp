#include <catch2/catch_test_macros.hpp>

#include <initializer_list>
#include <string>
#include <vector>

#include "../../cli/cli_args.h"

using adaptq::cli::IntParseStatus;
using adaptq::cli::ReplayOptions;
using adaptq::cli::ReplayParseStatus;
using adaptq::cli::parse_int_arg;
using adaptq::cli::parse_replay_args;

namespace {

/* Build a mutable argv array that mimics main()'s hand-off to cmd_replay. */
struct Argv {
    std::vector<std::string> storage;
    mutable std::vector<char *> pointers;

    Argv(std::initializer_list<const char *> args) {
        storage.reserve(args.size());
        for (const char *arg : args) storage.emplace_back(arg);
        pointers.reserve(storage.size());
        for (std::string &arg : storage) pointers.push_back(arg.data());
    }

    int argc() const { return static_cast<int>(pointers.size()); }
    char **argv() const { return pointers.data(); }
};

ReplayParseStatus parse(const Argv &argv, ReplayOptions &opts,
                        std::string &error) {
    return parse_replay_args(argv.argc(), argv.argv(), opts, error);
}

} /* namespace */

TEST_CASE("parse_int_arg accepts canonical decimal integers",
          "[cli][args]") {
    REQUIRE(parse_int_arg("0").ok());
    REQUIRE(parse_int_arg("0").value == 0);

    REQUIRE(parse_int_arg("42").ok());
    REQUIRE(parse_int_arg("42").value == 42);

    REQUIRE(parse_int_arg("+42").ok());
    REQUIRE(parse_int_arg("+42").value == 42);

    REQUIRE(parse_int_arg("-42").ok());
    REQUIRE(parse_int_arg("-42").value == -42);

    /* Leading zeroes are still base-10, not octal. */
    REQUIRE(parse_int_arg("007").ok());
    REQUIRE(parse_int_arg("007").value == 7);
}

TEST_CASE("parse_int_arg rejects malformed values", "[cli][args]") {
    REQUIRE(parse_int_arg("").status == IntParseStatus::empty);
    REQUIRE(parse_int_arg("+").status == IntParseStatus::invalid);
    REQUIRE(parse_int_arg("-").status == IntParseStatus::invalid);
    REQUIRE(parse_int_arg(" 12").status == IntParseStatus::invalid);
    REQUIRE(parse_int_arg("12 ").status == IntParseStatus::invalid);
    REQUIRE(parse_int_arg("1 2").status == IntParseStatus::invalid);
    REQUIRE(parse_int_arg("12abc").status == IntParseStatus::invalid);
    REQUIRE(parse_int_arg("abc").status == IntParseStatus::invalid);
    REQUIRE(parse_int_arg("0x10").status == IntParseStatus::invalid);
    REQUIRE(parse_int_arg("1e3").status == IntParseStatus::invalid);
    REQUIRE(parse_int_arg("1.5").status == IntParseStatus::invalid);
    REQUIRE(parse_int_arg("--1").status == IntParseStatus::invalid);
}

TEST_CASE("parse_int_arg enforces the requested window", "[cli][args]") {
    REQUIRE(parse_int_arg("10", 10, 20).ok());
    REQUIRE(parse_int_arg("20", 10, 20).ok());
    REQUIRE(parse_int_arg("9", 10, 20).status == IntParseStatus::out_of_range);
    REQUIRE(parse_int_arg("21", 10, 20).status == IntParseStatus::out_of_range);

    /* A syntactically valid value outside long long overflows cleanly. */
    REQUIRE(parse_int_arg("99999999999999999999999999").status ==
            IntParseStatus::out_of_range);

    /* Negative numbers are rejected when the caller demands a non-negative
     * window, which is how --from-token and --capacity are validated. */
    REQUIRE(parse_int_arg("-1", 0, 100).status == IntParseStatus::out_of_range);
}

TEST_CASE("parse_replay_args applies defaults", "[cli][args]") {
    Argv argv{"session.aqss"};
    ReplayOptions opts;
    std::string error;

    REQUIRE(parse(argv, opts, error) == ReplayParseStatus::ok);
    REQUIRE(opts.snapshot_path == "session.aqss");
    REQUIRE(opts.strategy.empty());
    REQUIRE(opts.output_path.empty());
    REQUIRE(opts.format == "json");
    REQUIRE(opts.from_token == -1);
    REQUIRE(opts.capacity == -1);
    REQUIRE_FALSE(opts.collect_metrics);
    REQUIRE_FALSE(opts.summary_json);
    REQUIRE(error.empty());
}

TEST_CASE("parse_replay_args accepts every documented option",
          "[cli][args]") {
    Argv argv{"session.aqss", "--strategy", "har_fixed",
              "--from-token", "12", "--capacity", "64",
              "--metrics", "--output", "report.csv",
              "--format", "csv", "--summary-json"};
    ReplayOptions opts;
    std::string error;

    REQUIRE(parse(argv, opts, error) == ReplayParseStatus::ok);
    REQUIRE(opts.snapshot_path == "session.aqss");
    REQUIRE(opts.strategy == "har_fixed");
    REQUIRE(opts.from_token == 12);
    REQUIRE(opts.capacity == 64);
    REQUIRE(opts.collect_metrics);
    REQUIRE(opts.output_path == "report.csv");
    REQUIRE(opts.format == "csv");
    REQUIRE(opts.summary_json);
}

TEST_CASE("parse_replay_args reports a missing option value",
          "[cli][args]") {
    const char *value_options[] = {"--strategy", "--from-token", "--capacity",
                                   "--output", "--format"};

    for (const char *option : value_options) {
        Argv argv{"session.aqss", option};
        ReplayOptions opts;
        std::string error;

        REQUIRE(parse(argv, opts, error) == ReplayParseStatus::error);
        REQUIRE(error.find(option) != std::string::npos);
        REQUIRE(error.find("requires a value") != std::string::npos);
    }
}

TEST_CASE("parse_replay_args rejects unknown and positional arguments",
          "[cli][args]") {
    {
        Argv argv{"session.aqss", "--fromtoken", "4"};
        ReplayOptions opts;
        std::string error;

        REQUIRE(parse(argv, opts, error) == ReplayParseStatus::error);
        REQUIRE(error.find("--fromtoken") != std::string::npos);
        REQUIRE(error.find("unknown option") != std::string::npos);
    }

    {
        Argv argv{"session.aqss", "extra.aqss"};
        ReplayOptions opts;
        std::string error;

        REQUIRE(parse(argv, opts, error) == ReplayParseStatus::error);
        REQUIRE(error.find("unexpected positional argument") != std::string::npos);
    }
}

TEST_CASE("parse_replay_args rejects non-numeric token counts",
          "[cli][args]") {
    const char *bad_values[] = {"", "abc", "12abc", " 12", "1.5", "0x10"};

    for (const char *bad : bad_values) {
        Argv argv{"session.aqss", "--from-token", bad};
        ReplayOptions opts;
        std::string error;

        REQUIRE(parse(argv, opts, error) == ReplayParseStatus::error);
        REQUIRE(error.find("--from-token") != std::string::npos);
    }

    {
        Argv argv{"session.aqss", "--capacity", "0"};
        ReplayOptions opts;
        std::string error;

        REQUIRE(parse(argv, opts, error) == ReplayParseStatus::error);
        REQUIRE(error.find("--capacity") != std::string::npos);
    }

    {
        Argv argv{"session.aqss", "--from-token", "-1"};
        ReplayOptions opts;
        std::string error;

        REQUIRE(parse(argv, opts, error) == ReplayParseStatus::error);
    }
}

TEST_CASE("parse_replay_args rejects unsupported formats", "[cli][args]") {
    Argv argv{"session.aqss", "--format", "yaml"};
    ReplayOptions opts;
    std::string error;

    REQUIRE(parse(argv, opts, error) == ReplayParseStatus::error);
    REQUIRE(error.find("yaml") != std::string::npos);
}

TEST_CASE("parse_replay_args leaves output untouched on failure",
          "[cli][args]") {
    ReplayOptions opts;
    opts.snapshot_path = "keep.aqss";
    opts.format = "md";
    std::string error;

    Argv argv{"session.aqss", "--capacity", "not-a-number"};
    REQUIRE(parse(argv, opts, error) == ReplayParseStatus::error);
    REQUIRE(opts.snapshot_path == "keep.aqss");
    REQUIRE(opts.format == "md");
}
