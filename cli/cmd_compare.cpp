#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "../runtime/runtime_context.h"
#include "../replay/session_snapshot.h"
#include "../replay/replay_engine.h"

/* -------------------------------------------------------------------------
 * cli/cmd_compare.cpp — adaptq compare <snapshot_file> --strategies A,B[,C]
 *
 * Usage:
 *   adaptq compare <snapshot.aqss>
 *     --strategies har_fixed,fp_passthrough
 *     [--format json|csv|md|tex]
 *     [--output <file>]
 *     [--metrics latency,quality,memory,bits]
 *     [--summary-json]
 *
 * Replays the same token log with each named strategy and reports the
 * per-strategy metrics in the requested format.
 * ----------------------------------------------------------------------- */

namespace adaptq {

/* =========================================================================
 * Metrics selector
 * ========================================================================= */
struct CompareRow {
    std::string strategy_name;
    double wall_time_ms;
    double avg_latency_us;
    double avg_quality;
    double avg_bits_per_dim;
    int    n_tokens;
};

static CompareRow summarize(const ReplayReport &r) {
    CompareRow row;
    row.strategy_name = r.strategy_name;
    row.wall_time_ms  = r.wall_time_ms;
    row.n_tokens      = r.n_tokens_replayed;

    double sum_lat = 0.0, sum_q = 0.0, sum_bits = 0.0;
    int    cnt_lat = 0,   cnt_q = 0,   cnt_bits = 0;

    for (const auto &m : r.metrics) {
        if (m.latency_us >= 0.f) { sum_lat  += m.latency_us;    ++cnt_lat;  }
        if (m.quality    >= 0.f) { sum_q    += m.quality;       ++cnt_q;    }
        if (m.avg_bits_per_dim >= 0.f) { sum_bits += m.avg_bits_per_dim; ++cnt_bits; }
    }

    row.avg_latency_us  = cnt_lat  ? sum_lat  / cnt_lat  : -1.0;
    row.avg_quality     = cnt_q    ? sum_q    / cnt_q    : -1.0;
    row.avg_bits_per_dim = cnt_bits ? sum_bits / cnt_bits : -1.0;
    return row;
}

/* =========================================================================
 * Formatters
 * ========================================================================= */

static void cmp_json(const std::vector<CompareRow> &rows, std::ostream &out) {
    out << "[\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto &r = rows[i];
        out << "  {\n"
            << "    \"strategy\": \"" << r.strategy_name << "\",\n"
            << "    \"n_tokens\": " << r.n_tokens << ",\n"
            << "    \"wall_time_ms\": " << r.wall_time_ms << ",\n"
            << "    \"avg_latency_us\": " << r.avg_latency_us << ",\n"
            << "    \"avg_quality\": " << r.avg_quality << ",\n"
            << "    \"avg_bits_per_dim\": " << r.avg_bits_per_dim << "\n"
            << "  }";
        if (i + 1 < rows.size()) out << ",";
        out << "\n";
    }
    out << "]\n";
}

static void cmp_csv(const std::vector<CompareRow> &rows, std::ostream &out) {
    out << "strategy,n_tokens,wall_time_ms,avg_latency_us,avg_quality,avg_bits_per_dim\n";
    for (const auto &r : rows) {
        out << r.strategy_name << ","
            << r.n_tokens << ","
            << r.wall_time_ms << ","
            << r.avg_latency_us << ","
            << r.avg_quality << ","
            << r.avg_bits_per_dim << "\n";
    }
}

static void cmp_md(const std::vector<CompareRow> &rows, std::ostream &out) {
    out << "# Strategy Comparison\n\n"
        << "| Strategy | Tokens | Wall (ms) | Latency µs | Quality | Bits/Dim |\n"
        << "|---|---|---|---|---|---|\n";
    for (const auto &r : rows) {
        out << "| `" << r.strategy_name << "`"
            << " | " << r.n_tokens
            << " | " << std::fixed << std::setprecision(2) << r.wall_time_ms
            << " | " << r.avg_latency_us
            << " | " << r.avg_quality
            << " | " << r.avg_bits_per_dim
            << " |\n";
    }
}

static void cmp_tex(const std::vector<CompareRow> &rows, std::ostream &out) {
    out << "\\begin{table}[h]\n"
        << "\\centering\n"
        << "\\begin{tabular}{lrrrrr}\n"
        << "\\hline\n"
        << "Strategy & Tokens & Wall (ms) & Latency (\\textmu s) & Quality & Bits/Dim \\\\\n"
        << "\\hline\n";
    for (const auto &r : rows) {
        out << r.strategy_name
            << " & " << r.n_tokens
            << " & " << std::fixed << std::setprecision(2) << r.wall_time_ms
            << " & " << r.avg_latency_us
            << " & " << r.avg_quality
            << " & " << r.avg_bits_per_dim
            << " \\\\\n";
    }
    out << "\\hline\n"
        << "\\end{tabular}\n"
        << "\\caption{AdapTQ Strategy Comparison}\n"
        << "\\end{table}\n";
}

/* =========================================================================
 * Strategy name tokenizer
 * ========================================================================= */
static std::vector<std::string> split_csv(const std::string &s) {
    std::vector<std::string> parts;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) parts.push_back(token);
    }
    return parts;
}

/* =========================================================================
 * cmd_compare entry point
 * ========================================================================= */

int cmd_compare(int argc, char **argv) {
    if (argc < 1) {
        std::cerr << "Usage: adaptq compare <snapshot.aqss> --strategies A,B[,C]\n"
                     "Options:\n"
                     "  --strategies har_fixed,fp_passthrough\n"
                     "  --format json|csv|md|tex\n"
                     "  --output <file>\n"
                     "  --summary-json\n";
        return 1;
    }

    std::string snap_path   = argv[0];
    std::string strategies_arg;
    std::string format      = "json";
    std::string output_path;
    bool        summary_json = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--strategies") == 0 && i + 1 < argc) {
            strategies_arg = argv[++i];
        } else if (strcmp(argv[i], "--format") == 0 && i + 1 < argc) {
            format = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--summary-json") == 0) {
            summary_json = true;
        }
    }

    if (strategies_arg.empty()) {
        std::cerr << "ERROR: --strategies is required. Example: --strategies har_fixed,fp_passthrough\n";
        return 1;
    }

    std::vector<std::string> strat_names = split_csv(strategies_arg);
    if (strat_names.empty()) {
        std::cerr << "ERROR: no strategies specified.\n";
        return 1;
    }

    /* Load snapshot. */
    SessionSnapshot snap;
    try {
        snap = SessionSnapshot::load(snap_path);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    if (!snap.has_token_log()) {
        std::cerr << "ERROR: snapshot has no token log. Re-capture with log_tokens=true.\n";
        return 1;
    }

    /* Build config from snapshot. */
    RuntimeContextConfig cfg;
    cfg.n_layers   = snap.n_layers();
    cfg.n_heads    = snap.n_heads();
    cfg.dim        = snap.dim();
    cfg.bits       = snap.bits();
    cfg.log_tokens = false;

    /* Run each strategy. */
    ReplayEngine engine(/*collect_metrics=*/true);
    std::vector<CompareRow> rows;

    for (const auto &name : strat_names) {
        std::cerr << "INFO: replaying with strategy '" << name << "' ...\n";
        try {
            ReplayReport r = engine.replay_with(snap, name, cfg);
            rows.push_back(summarize(r));
            std::cerr << "  done. " << r.n_tokens_replayed
                      << " tokens in " << r.wall_time_ms << " ms\n";
        } catch (const std::exception &e) {
            std::cerr << "ERROR (strategy=" << name << "): " << e.what() << "\n";
            return 1;
        }
    }

    /* Write output. */
    std::ostream *pout = &std::cout;
    std::ofstream fout;
    if (!output_path.empty()) {
        fout.open(output_path);
        if (!fout) {
            std::cerr << "ERROR: cannot write to " << output_path << "\n";
            return 1;
        }
        pout = &fout;
    }

    if (format == "csv")
        cmp_csv(rows, *pout);
    else if (format == "md")
        cmp_md(rows, *pout);
    else if (format == "tex")
        cmp_tex(rows, *pout);
    else
        cmp_json(rows, *pout);

    /* The Python compare API requests the human-readable artifact in the
     * requested format while also needing structured rows. Keep the two
     * channels separate so each strategy is replayed only once. */
    if (summary_json)
        cmp_json(rows, std::cout);

    return 0;
}

} /* namespace adaptq */
