"""The gate must pass the committed release and fail every way a bad commit
can look: an unfair policy, a policy that skips a group, a model swapped
without new predictions, and a calibrator that no longer matches."""
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gate  # noqa: E402


@pytest.fixture
def repo(tmp_path):
    for d in ("models", "runs"):
        shutil.copytree(ROOT / d, tmp_path / d)
    return tmp_path


def run(root, *extra):
    return gate.main(["--root", str(root), *extra])


def test_committed_release_passes(repo):
    assert run(repo) == 0


def test_unfair_policy_fails(repo):
    assert run(repo, "--policy", str(ROOT / "demo" / "unfair_commit_policy.json")) == 1


def test_policy_missing_a_group_fails(repo):
    p = repo / "partial.json"
    p.write_text('{"thresholds": {"Female|35 and over": 0.2, "Male|35 and over": 0.2}}')
    assert run(repo, "--policy", str(p)) == 1


def test_swapped_model_fails_even_with_old_predictions(repo):
    m = repo / "models" / "xgb.json"
    m.write_text(m.read_text() + " ")  # any change at all, the model still loads
    assert run(repo) == 1


def test_calibrator_mismatch_fails(repo):
    import pickle

    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip").fit(np.linspace(0, 1, 50), np.linspace(0, 0.5, 50))
    pickle.dump(iso, open(repo / "models" / "calibrator.pkl", "wb"))
    # even after re-recording the manifest, G2 catches the mismatch
    from manifest import write_manifest
    write_manifest(repo)
    assert run(repo) == 1


def test_expected_profit_policy_missing_a_group_fails(repo):
    import json
    pol = json.loads((ROOT / "models" / "fair_policy.json").read_text())
    if pol.get("type") != "expected_profit":
        pytest.skip("deployed policy is not expected-profit")
    pol["offsets"].pop("Male|Under 35")
    p = repo / "partial_profit.json"
    p.write_text(json.dumps(pol))
    assert run(repo, "--policy", str(p)) == 1


def test_gaps_are_gated_on_the_upper_bound_not_the_point():
    """A gap of 4 points measured on a small sample passes on its point value
    but its 95% interval reaches past 5 points, so the gate must fail it."""
    import numpy as np
    rng = np.random.default_rng(0)
    n = 1200                                   # 600 per group, 552 repayers each
    g = np.repeat(["A", "B"], n // 2)
    age = np.tile(["Under 35", "35 and over"], n // 2)
    y = np.tile(np.r_[np.zeros(552, int), np.ones(48, int)], 2)
    p = np.clip(0.08 + rng.normal(0, 0.03, n), 0.001, 0.999)
    rank = np.tile(np.arange(600), 2)          # repayers first within each group
    approve = np.where(g == "A", rank < round(0.90 * 552), rank < round(0.86 * 552))  # gap 4.0 points
    checks = gate.requirement_checks(y, p, approve, g, age, g)
    nfr3 = next(c for c in checks if "gender" in c[0])
    point, high = nfr3[1]["point"], nfr3[1]["high"]
    assert point <= 0.05 < high and nfr3[2] is False
