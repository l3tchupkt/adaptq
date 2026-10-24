#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "../runtime/runtime_context.h"
#include "../replay/session_snapshot.h"
#include "../replay/replay_engine.h"
#include "cli_args.h"

/* -------------------------------------------------------------------------
 * cli/cmd_replay.cpp — adaptq replay <snapshot_file> [options]
 *
 * Usage:
 *   adaptq replay <snapshot.aqss>
 *     [--strategy har_fixed|fp_passthrough]
 *     [--from-token N]
 *     [--capacity N]
 *     [--metrics]
 *     [--output <file>]
 *     [--format json|csv|md|tex]
 *     [--summary-json]
 *
 * Default:
 *   Replays all tokens using the same strategy configuration as the snapshot.
 *   Outputs a JSON ReplayReport to stdout.
 * ----------------------------------------------------------------------- */

namespace adaptq {

/* make_contiguous() is defined in runtime_context.cpp (same link unit). */
extern IStorageBackend *make_contiguous();

/* =========================================================================
 * Output formatters
 * ========================================================================= */

static void format_json(const ReplayReport &r, const SessionSnapshot &snap,
                        std::ostream &out) {
    out << "{\n"
        << "  \"strategy\": \"" << r.strategy_name << "\",\n"
        << "  \"n_tokens_replayed\": " << r.n_tokens_replayed << ",\n"
        << "  \"wall_time_ms\": " << r.wall_time_ms << ",\n"
        << "  \"snapshot_n_tokens\": " << snap.n_tokens() << ",\n"
        << "  \"n_layers\": " << snap.n_layers() << ",\n"
        << "  \"n_heads\": " << snap.n_heads() << ",\n"
        << "  \"dim\": " << snap.dim() << "\n";

    if (!r.metrics.empty()) {
        out << "  ,\"metrics\": [\n";
        for (size_t i = 0; i < r.metrics.size(); ++i) {
            const auto &m = r.metrics[i];
            out << "    {\"layer\":" << m.layer
                << ",\"head\":" << m.head
                << ",\"n_tokens_used\":" << m.n_tokens_used
                << ",\"latency_us\":" << m.latency_us
                << ",\"quality\":" << m.quality
                << ",\"avg_bits_per_dim\":" << m.avg_bits_per_dim
                << "}";
            if (i + 1 < r.metrics.size()) out << ",";
            out << "\n";
        }
        out << "  ]\n";
    }
    out << "}\n";
}

static void format_csv(const ReplayReport &r, std::ostream &out) {
    out << "strategy,n_tokens_replayed,wall_time_ms\n";
    out << r.strategy_name << "," << r.n_tokens_replayed << "," << r.wall_time_ms << "\n";

    if (!r.metrics.empty()) {
        out << "\nlayer,head,n_tokens_used,latency_us,quality,avg_bits_per_dim\n";
        for (const auto &m : r.metrics) {
            out << m.layer << "," << m.head << "," << m.n_tokens_used
                << "," << m.latency_us << "," << m.quality
                << "," << m.avg_bits_per_dim << "\n";
        }
    }
}

static void format_md(const ReplayReport &r, const SessionSnapshot &snap,
                      std::ostream &out) {
    out << "# Replay Report\n\n"
        << "| Field | Value |\n"
        << "|---|---|\n"
        << "| Strategy | `" << r.strategy_name << "` |\n"
        << "| Tokens Replayed | " << r.n_tokens_replayed << " |\n"
        << "| Snapshot Tokens | " << snap.n_tokens() << " |\n"
        << "| Wall Time | " << r.wall_time_ms << " ms |\n"
        << "| Layers | " << snap.n_layers() << " |\n"
        << "| Heads | " << snap.n_heads() << " |\n"
        << "| Dim | " << snap.dim() << " |\n"
        << "\n";

    if (!r.metrics.empty()) {
        out << "## Per-Token Metrics\n\n"
            << "| Layer | Head | N Tokens | Latency (µs) | Quality | Bits/Dim |\n"
            << "|---|---|---|---|---|---|\n";
        for (const auto &m : r.metrics) {
            out << "| " << m.layer << " | " << m.head
                << " | " << m.n_tokens_used
                << " | " << m.latency_us
                << " | " << m.quality
                << " | " << m.avg_bits_per_dim << " |\n";
        }
    }
}

static void format_tex(const ReplayReport &r, const SessionSnapshot &snap,
                       std::ostream &out) {
    out << "\\begin{table}[h]\n"
        << "\\centering\n"
        << "\\begin{tabular}{ll}\n"
        << "\\hline\n"
        << "Field & Value \\\\\n"
        << "\\hline\n"
        << "Strategy & " << r.strategy_name << " \\\\\n"
        << "Tokens Replayed & " << r.n_tokens_replayed << " \\\\\n"
        << "Snapshot Tokens & " << snap.n_tokens() << " \\\\\n"
        << "Wall Time & " << r.wall_time_ms << " ms \\\\\n"
        << "Layers & " << snap.n_layers() << " \\\\\n"
        << "Heads & " << snap.n_heads() << " \\\\\n"
        << "Dim & " << snap.dim() << " \\\\\n"
        << "\\hline\n"
        << "\\end{tabular}\n"
        << "\\caption{AdapTQ Replay Report}\n"
        << "\\end{table}\n";
}

/* =========================================================================
 * cmd_replay entry point
 * ========================================================================= */

int cmd_replay(int argc, char **argv) {
    if (argc < 1) {
        std::cerr << "Usage: adaptq replay <snapshot.aqss> [options]\n"
                     "Options:\n"
                     "  --strategy har_fixed|fp_passthrough\n"
                     "  --from-token N\n"
                     "  --capacity N\n"
                     "  --metrics\n"
                     "  --output <file>\n"
                     "  --format json|csv|md|tex\n"
                     "  --summary-json\n";
        return 1;
    }

    cli::ReplayOptions opts;
    std::string        parse_error;
    if (cli::parse_replay_args(argc, argv, opts, parse_error) !=
        cli::ReplayParseStatus::ok) {
        std::cerr << "ERROR: " << parse_error << "\n"
                  << "Run 'adaptq replay' without arguments to see usage.\n";
        return 1;
    }

    /* Load snapshot. */
    SessionSnapshot snap;
    try {
        snap = SessionSnapshot::load(opts.snapshot_path);
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    if (!snap.has_token_log()) {
        std::cerr << "ERROR: snapshot has no token log. Re-capture with log_tokens=true.\n";
        return 1;
    }

    if (opts.from_token >= 0 && opts.from_token > snap.n_tokens()) {
        std::cerr << "ERROR: --from-token " << opts.from_token
                  << " is outside [0, " << snap.n_tokens()
                  << "] for this snapshot\n";
        return 1;
    }

    /* Build RuntimeContext. */
    RuntimeContextConfig cfg;
    cfg.n_layers    = snap.n_layers();
    cfg.n_heads     = snap.n_heads();
    cfg.dim         = snap.dim();
    cfg.bits        = snap.bits();
    cfg.log_tokens  = false;
    if (opts.capacity > 0) {
        cfg.capacity = static_cast<int>(opts.capacity);
    } else {
        cfg.capacity = std::max(cfg.capacity, snap.n_tokens());
    }

    RuntimeContext ctx;
    if (opts.strategy.empty()) {
        ctx.init(cfg);
    } else {
        StrategyFactory sfn = strategy_factory_by_name(opts.strategy.c_str());
        if (!sfn) {
            std::cerr << "ERROR: unknown strategy '" << opts.strategy << "'\n"
                      << "Available: har_fixed, fp_passthrough\n";
            return 1;
        }
        /* Use internal make_contiguous via default init then re-init with factory. */
        ctx.init(cfg, sfn, make_contiguous);
    }

    ReplayEngine engine(opts.collect_metrics);
    ReplayReport report;

    try {
        if (opts.from_token >= 0) {
            /* Branch mode — warm-up then stop. */
            engine.branch(snap, ctx, static_cast<int>(opts.from_token));
            report.n_tokens_replayed = static_cast<int>(opts.from_token);
            report.strategy_name     = ctx.get_strategy(0, 0)
                                        ? ctx.get_strategy(0, 0)->name()
                                        : "unknown";
            std::cerr << "INFO: Context warmed up to token " << opts.from_token
                      << ". Ready for continued inference.\n";
        } else {
            report = engine.replay(snap, ctx);
        }
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    /* Write output. */
    std::ostream *pout = &std::cout;
    std::ofstream fout;
    if (!opts.output_path.empty()) {
        fout.open(opts.output_path);
        if (!fout) {
            std::cerr << "ERROR: cannot write to " << opts.output_path << "\n";
            return 1;
        }
        pout = &fout;
    }

    if (opts.format == "csv")
        format_csv(report, *pout);
    else if (opts.format == "md")
        format_md(report, snap, *pout);
    else if (opts.format == "tex")
        format_tex(report, snap, *pout);
    else
        format_json(report, snap, *pout);

    /* The Python replay API requests the human-readable artifact in the
     * requested format while also needing the structured result object. Keep
     * the two channels separate so the replay itself happens only once. */
    if (opts.summary_json)
        format_json(report, snap, std::cout);

    return 0;
}

} /* namespace adaptq */
