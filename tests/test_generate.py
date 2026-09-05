from reconpilot.db import init_db, session_scope
from reconpilot.generate import (
    SCENARIO_MIX,
    TOTAL_CASES,
    generate_cases,
    get_scenario_counts,
)
from reconpilot.models import Case, GroundTruthRow


def test_scenario_mix_totals():
    assert TOTAL_CASES >= 100
    assert sum(SCENARIO_MIX.values()) == TOTAL_CASES
    assert all(count > 0 for count in SCENARIO_MIX.values())


def test_generate_exact_scenario_distribution(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"

    monkeypatch.setenv(
        "RECONPILOT_DATABASE_URL",
        f"sqlite:///{db_path}",
    )

    # The project's current config may use a fixed database URL,
    # so initialize the normal DB for this smoke test.
    init_db()

    generate_cases(seed=42, replace_existing=True)

    counts = get_scenario_counts()

    assert counts == SCENARIO_MIX


def test_generated_case_count():
    generate_cases(seed=42, replace_existing=True)

    with session_scope() as session:
        case_count = session.query(Case).count()
        gt_count = session.query(GroundTruthRow).count()

    assert case_count == TOTAL_CASES
    assert gt_count == TOTAL_CASES


def test_ground_truth_is_separate_from_operational_case():
    generate_cases(seed=42, replace_existing=True)

    with session_scope() as session:
        case = session.query(Case).first()
        gt = (
            session.query(GroundTruthRow)
            .filter_by(case_id=case.case_id)
            .one()
        )

    assert case.case_id == gt.case_id
    assert gt.scenario_type in SCENARIO_MIX
    assert "expected_outcome" not in case.payload_json