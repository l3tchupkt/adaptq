/* Catch2 v3 — link against Catch2::Catch2WithMain */
#include "strategy_conformance.h"        /* run_conformance<T> template  */
#include "../../strategies/fp_passthrough.cpp"  /* compiled inline here only  */
#include "../../strategies/har_fixed.cpp"       /* compiled inline here only  */

/* -------------------------------------------------------------------------
 * tests/conformance/strategy_conformance.cpp
 *
 * Runs the IKVStrategy conformance suite against all built-in strategies.
 *
 * IMPORTANT: fp_passthrough.cpp and har_fixed.cpp are included here
 * directly (not via CMake source list). This is intentional — they are
 * not part of any library and exist only as plugin templates. Including
 * them here makes this test the single TU that compiles and exercises them.
 *
 * To add a new strategy:
 *   #include "../../strategies/<name>.cpp"
 *   TEST_CASE("<Name> conformance", "[conformance]") {
 *       adaptq::run_conformance<adaptq::<Name>Strategy>("<Name>");
 *   }
 * ----------------------------------------------------------------------- */

TEST_CASE("FPPassthroughStrategy conformance", "[conformance]") {
    adaptq::run_conformance<adaptq::FPPassthroughStrategy>("FPPassthroughStrategy");
}

TEST_CASE("HARFixedStrategy conformance", "[conformance]") {
    adaptq::run_conformance<adaptq::HARFixedStrategy>("HARFixedStrategy");
}
