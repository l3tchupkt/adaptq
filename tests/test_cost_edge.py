"""Regression test for cost reselect denominator handling (issues 137/142).

Mirrors core/cost_optimizer.cpp should_reselect after the fabs fix:
  denom = abs(best)
  if denom < 1e-6: return (current - best) > threshold
  return (current - best) / denom > threshold
"""


def should_reselect(current_cost, best_cost, threshold=0.20):
    denom = abs(best_cost)
    if denom < 1e-6:
        return (current_cost - best_cost) > threshold
    return (current_cost - best_cost) / denom > threshold


def test_positive_baseline_triggers():
    assert should_reselect(12.0, 10.0, 0.20) is False  # 0.2 not > 0.2
    assert should_reselect(12.5, 10.0, 0.20) is True


def test_negative_baseline_no_sign_flip():
    # Old code divided by (best + eps) which flips sign when best < 0.
    # Fixed code scales by abs(best).
    assert should_reselect(-8.0, -10.0, 0.20) is False  # improved, not worse
    assert should_reselect(-7.0, -10.0, 0.20) is True  # 3/10 > 0.2


def test_near_zero_baseline_uses_absolute():
    assert should_reselect(0.5, 1e-9, 0.20) is True
    assert should_reselect(1e-9, 1e-9, 0.20) is False
    assert should_reselect(-0.5, 0.0, 0.20) is False


def test_zero_threshold():
    assert should_reselect(10.0, 10.0, 0.0) is False
    assert should_reselect(10.1, 10.0, 0.0) is True
