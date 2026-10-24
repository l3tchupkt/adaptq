#pragma once
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <string>

/* -------------------------------------------------------------------------
 * cli/cli_args.h — shared command-line parsing helpers
 *
 * The replay subcommand used to feed option values straight into std::atoi,
 * which silently turned "12abc" into 12, an empty value into 0, and values
 * larger than INT_MAX into an implementation-defined result. Unknown options
 * and options with a missing value were ignored entirely, so a typo such as
 * "--fromtoken 4" produced a full replay instead of a branch. This header
 * provides strict, testable parsing so the command can fail with a useful
 * message instead of quietly doing the wrong thing.
 * ----------------------------------------------------------------------- */

namespace adaptq {
namespace cli {

enum class IntParseStatus {
    ok,
    empty,
    invalid,      /* non-decimal character, lone sign, embedded whitespace */
    out_of_range  /* overflow or outside the caller's [min, max] window */
};

struct IntParseResult {
    IntParseStatus status = IntParseStatus::empty;
    long long      value  = 0;

    bool ok() const noexcept { return status == IntParseStatus::ok; }
};

/**
 * Parse a strict base-10 integer.
 *
 * Accepted grammar: optional '+'/'-' followed by one or more decimal digits.
 * Everything else is rejected: an empty string, a lone sign, surrounding or
 * embedded whitespace, hexadecimal, exponent notation, and trailing garbage.
 * When min_value/max_value are supplied the parsed value must fall inside
 * that inclusive window.
 */
inline IntParseResult parse_int_arg(const std::string &text,
                                    long long min_value = LLONG_MIN,
                                    long long max_value = LLONG_MAX) {
    IntParseResult result;

    if (text.empty()) {
        result.status = IntParseStatus::empty;
        return result;
    }

    std::size_t first_digit = 0;
    if (text[0] == '+' || text[0] == '-') first_digit = 1;
    if (first_digit == text.size()) {
        result.status = IntParseStatus::invalid;
        return result;
    }
    for (std::size_t i = first_digit; i < text.size(); ++i) {
        if (text[i] < '0' || text[i] > '9') {
            result.status = IntParseStatus::invalid;
            return result;
        }
    }

    errno = 0;
    char *end = nullptr;
    const long long parsed = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end == nullptr || *end != '\0') {
        result.status = IntParseStatus::out_of_range;
        return result;
    }
    if (parsed < min_value || parsed > max_value) {
        result.status = IntParseStatus::out_of_range;
        return result;
    }

    result.status = IntParseStatus::ok;
    result.value  = parsed;
    return result;
}

inline const char *int_parse_status_message(IntParseStatus status) {
    switch (status) {
        case IntParseStatus::empty:
            return "value is empty";
        case IntParseStatus::invalid:
            return "expected a decimal integer";
        case IntParseStatus::out_of_range:
            return "value is out of range";
        case IntParseStatus::ok:
            break;
    }
    return "value accepted";
}

/* ---- replay subcommand options ---------------------------------------- */

struct ReplayOptions {
    std::string snapshot_path;
    std::string strategy;
    std::string output_path;
    std::string format = "json";
    long long   from_token = -1;  /* -1 means "full replay" */
    long long   capacity   = -1;  /* -1 means "derive from snapshot" */
    bool        collect_metrics = false;
    bool        summary_json    = false;
};

enum class ReplayParseStatus { ok, error };

/**
 * Parse and validate the argument vector handed to cmd_replay(). argv[0] is
 * the snapshot path, matching main()'s call site (argv + 2).
 *
 * On failure the function leaves opts untouched and stores a human-readable
 * message in error, without a trailing newline or "ERROR:" prefix.
 */
inline ReplayParseStatus parse_replay_args(int argc, char **argv,
                                           ReplayOptions &opts,
                                           std::string &error) {
    if (argc < 1 || argv == nullptr || argv[0] == nullptr ||
        argv[0][0] == '\0') {
        error = "missing snapshot path";
        return ReplayParseStatus::error;
    }

    ReplayOptions parsed;
    parsed.snapshot_path = argv[0];

    auto take_string = [&](int &i, const char *name, std::string &dst) -> bool {
        if (i + 1 >= argc) {
            error = std::string("option '") + name + "' requires a value";
            return false;
        }
        dst = argv[++i];
        return true;
    };

    auto take_integer = [&](int &i, const char *name, long long min_value,
                            long long max_value, long long &dst) -> bool {
        if (i + 1 >= argc) {
            error = std::string("option '") + name + "' requires a value";
            return false;
        }
        const std::string raw = argv[++i];
        const IntParseResult res =
            parse_int_arg(raw, min_value, max_value);
        if (!res.ok()) {
            error = std::string("invalid value '") + raw + "' for option '" +
                    name + "': " + int_parse_status_message(res.status);
            return false;
        }
        dst = res.value;
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--strategy") {
            if (!take_string(i, "--strategy", parsed.strategy)) {
                return ReplayParseStatus::error;
            }
        } else if (arg == "--from-token") {
            if (!take_integer(i, "--from-token", 0, INT_MAX,
                              parsed.from_token)) {
                return ReplayParseStatus::error;
            }
        } else if (arg == "--capacity") {
            if (!take_integer(i, "--capacity", 1, INT_MAX,
                              parsed.capacity)) {
                return ReplayParseStatus::error;
            }
        } else if (arg == "--output") {
            if (!take_string(i, "--output", parsed.output_path)) {
                return ReplayParseStatus::error;
            }
        } else if (arg == "--format") {
            if (!take_string(i, "--format", parsed.format)) {
                return ReplayParseStatus::error;
            }
        } else if (arg == "--metrics") {
            parsed.collect_metrics = true;
        } else if (arg == "--summary-json") {
            parsed.summary_json = true;
        } else if (!arg.empty() && arg[0] == '-') {
            error = "unknown option '" + arg + "'";
            return ReplayParseStatus::error;
        } else {
            error = "unexpected positional argument '" + arg + "'";
            return ReplayParseStatus::error;
        }
    }

    if (parsed.format != "json" && parsed.format != "csv" &&
        parsed.format != "md" && parsed.format != "tex") {
        error = "unsupported output format '" + parsed.format +
                "' (expected json, csv, md, or tex)";
        return ReplayParseStatus::error;
    }

    opts = parsed;
    return ReplayParseStatus::ok;
}

} /* namespace cli */
} /* namespace adaptq */
