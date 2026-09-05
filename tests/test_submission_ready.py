import json
from pathlib import Path

from reconpilot.ai_investigator import investigate_locally
from reconpilot.benchmark import run_benchmark
from reconpilot.candidates import Candidate
from reconpilot.generate import SCENARIO_MIX
from validate import validate_payload


def test_local_evaluator_is_explicit_and_safe():
    candidates=[Candidate("BANK-1","bank_txn",0.95,("amount_exact",))]
    result=investigate_locally({"bank_txn_id":"BANK-1"},candidates)
    assert result.success
    assert result.investigation is not None
    assert result.investigation.selected_candidate_id == "BANK-1"


def test_invalid_amount_is_classified_as_invalid_record():
    errors=validate_payload({"invoice":{"amount":"100"},"payment":{"amount":"NOT_A_NUMBER"},"settlement":None,"bank_txn":{"amount":"100"}})
    codes={x["code"] for x in errors}
    assert "INVALID_RECORD" in codes
    assert "MISSING_RECORD" in codes


def test_benchmark_writes_reproducible_artifacts(tmp_path):
    report=run_benchmark(seed=7,ai_mode="local",output_dir=tmp_path)
    assert report["baseline"]["total_cases"] == sum(SCENARIO_MIX.values())
    assert report["reconpilot"]["metrics"]["false_auto_match_count"] == 0
    assert report["reconpilot"]["candidate_quality"]["recall_at_3"] == 1.0
    for name in ("run_seed_7.json","metrics.json","scenario_metrics.json","exceptions.json"):
        assert (tmp_path/name).exists()
        json.loads((tmp_path/name).read_text())
