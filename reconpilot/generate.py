from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from faker import Faker
from sqlalchemy import delete

from reconpilot.db import session_scope, init_db
from reconpilot.models import Case, GroundTruthRow


# ============================================================
# CONTROLLED DATASET
# ============================================================

SCENARIO_MIX: dict[str, int] = {
    "exact_match": 20,
    "fee_adjusted_match": 15,
    "delayed_settlement": 10,
    "fuzzy_reference": 10,
    "duplicate_candidate": 10,
    "partial_payment": 8,
    "missing_counterparty": 8,
    "corrupt_record": 5,
    "no_valid_match": 8,
    "ambiguous_match": 10,
}

TOTAL_CASES = sum(SCENARIO_MIX.values())


# ============================================================
# HELPERS
# ============================================================

fake = Faker("en_IN")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def money(value: float | Decimal) -> str:
    return f"{Decimal(str(value)):.2f}"


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def make_date(days_ago: int = 0) -> str:
    dt = utcnow() - timedelta(days=days_ago)
    return dt.isoformat()


def base_case(case_id: str, scenario: str) -> dict[str, Any]:
    invoice_id = make_id("INV")
    payment_id = make_id("PAY")
    settlement_id = make_id("SET")
    bank_txn_id = make_id("BANK")

    amount = Decimal(random.choice([
        "1250.00",
        "2499.00",
        "4999.00",
        "7500.00",
        "12500.00",
        "24999.00",
    ]))

    invoice = {
        "invoice_id": invoice_id,
        "customer_name": fake.company(),
        "amount": money(amount),
        "currency": "INR",
        "issued_at": make_date(random.randint(2, 20)),
        "reference": invoice_id,
    }

    payment = {
        "payment_id": payment_id,
        "invoice_id": invoice_id,
        "amount": money(amount),
        "currency": "INR",
        "paid_at": make_date(random.randint(1, 10)),
        "utr": f"UTR{random.randint(10**10, 10**11 - 1)}",
        "method": random.choice(["UPI", "CARD", "NETBANKING"]),
        "status": "captured",
    }

    settlement = {
        "settlement_id": settlement_id,
        "payment_id": payment_id,
        "gross_amount": money(amount),
        "fee_amount": "0.00",
        "net_amount": money(amount),
        "currency": "INR",
        "settled_at": make_date(random.randint(0, 7)),
        "utr": payment["utr"],
    }

    bank = {
        "bank_txn_id": bank_txn_id,
        "amount": money(amount),
        "currency": "INR",
        "booked_at": make_date(random.randint(0, 7)),
        "narration": f"Payment received {invoice_id}",
        "utr": payment["utr"],
        "counterparty": invoice["customer_name"],
    }

    return {
        "invoice": invoice,
        "payment": payment,
        "settlement": settlement,
        "bank_txn": bank,
    }


# ============================================================
# SCENARIOS
# ============================================================

def scenario_exact_match(case_id: str) -> tuple[dict, dict]:
    data = base_case(case_id, "exact_match")

    gt = {
        "case_id": case_id,
        "scenario_type": "exact_match",
        "expected_outcome": "match",
        "expected_invoice_id": data["invoice"]["invoice_id"],
        "expected_payment_id": data["payment"]["payment_id"],
        "expected_settlement_id": data["settlement"]["settlement_id"],
        "expected_bank_txn_id": data["bank_txn"]["bank_txn_id"],
    }

    return data, gt


def scenario_fee_adjusted_match(case_id: str) -> tuple[dict, dict]:
    data = base_case(case_id, "fee_adjusted_match")

    gross = Decimal(data["payment"]["amount"])
    fee = (gross * Decimal("0.02")).quantize(Decimal("0.01"))
    net = gross - fee

    data["settlement"]["gross_amount"] = money(gross)
    data["settlement"]["fee_amount"] = money(fee)
    data["settlement"]["net_amount"] = money(net)

    data["bank_txn"]["amount"] = money(net)

    gt = {
        "case_id": case_id,
        "scenario_type": "fee_adjusted_match",
        "expected_outcome": "match",
        "expected_invoice_id": data["invoice"]["invoice_id"],
        "expected_payment_id": data["payment"]["payment_id"],
        "expected_settlement_id": data["settlement"]["settlement_id"],
        "expected_bank_txn_id": data["bank_txn"]["bank_txn_id"],
    }

    return data, gt


def scenario_delayed_settlement(case_id: str) -> tuple[dict, dict]:
    data = base_case(case_id, "delayed_settlement")

    data["settlement"]["settled_at"] = (
        utcnow() - timedelta(days=18)
    ).isoformat()

    data["bank_txn"]["booked_at"] = (
        utcnow() - timedelta(days=17)
    ).isoformat()

    gt = {
        "case_id": case_id,
        "scenario_type": "delayed_settlement",
        "expected_outcome": "match",
        "expected_invoice_id": data["invoice"]["invoice_id"],
        "expected_payment_id": data["payment"]["payment_id"],
        "expected_settlement_id": data["settlement"]["settlement_id"],
        "expected_bank_txn_id": data["bank_txn"]["bank_txn_id"],
    }

    return data, gt


def scenario_fuzzy_reference(case_id: str) -> tuple[dict, dict]:
    data = base_case(case_id, "fuzzy_reference")

    invoice_id = data["invoice"]["invoice_id"]

    data["bank_txn"]["narration"] = (
        f"Incoming payment ref {invoice_id[:6]} "
        f"{invoice_id[-4:]}"
    )

    data["payment"]["utr"] = data["payment"]["utr"][:-2] + "XX"

    gt = {
        "case_id": case_id,
        "scenario_type": "fuzzy_reference",
        "expected_outcome": "match",
        "expected_invoice_id": data["invoice"]["invoice_id"],
        "expected_payment_id": data["payment"]["payment_id"],
        "expected_settlement_id": data["settlement"]["settlement_id"],
        "expected_bank_txn_id": data["bank_txn"]["bank_txn_id"],
    }

    return data, gt


def scenario_duplicate_candidate(case_id: str) -> tuple[dict, dict]:
    data = base_case(case_id, "duplicate_candidate")

    winner = data["bank_txn"]["bank_txn_id"]

    duplicate = dict(data["bank_txn"])
    # Keep the real transaction first in deterministic tie-breaking.
    # It remains indistinguishable on amount/date/UTR, so the case is
    # still intentionally ambiguous until AI selects a candidate.
    duplicate["bank_txn_id"] = f"BANK_DUP_{uuid.uuid4().hex[:8]}"
    duplicate["narration"] = "Incoming transfer duplicate"
    duplicate["utr"] = f"DUP-{uuid.uuid4().hex[:10]}"

    data["bank_txn_duplicates"] = [duplicate]

    gt = {
        "case_id": case_id,
        "scenario_type": "duplicate_candidate",
        "expected_outcome": "match",
        "expected_invoice_id": data["invoice"]["invoice_id"],
        "expected_payment_id": data["payment"]["payment_id"],
        "expected_settlement_id": data["settlement"]["settlement_id"],
        "expected_bank_txn_id": winner,
    }

    return data, gt


def scenario_partial_payment(case_id: str) -> tuple[dict, dict]:
    data = base_case(case_id, "partial_payment")

    invoice_amount = Decimal(data["invoice"]["amount"])
    partial = (invoice_amount * Decimal("0.40")).quantize(Decimal("0.01"))

    data["payment"]["amount"] = money(partial)
    data["settlement"]["gross_amount"] = money(partial)
    data["settlement"]["net_amount"] = money(partial)
    data["bank_txn"]["amount"] = money(partial)

    gt = {
        "case_id": case_id,
        "scenario_type": "partial_payment",
        "expected_outcome": "exception",
        "expected_invoice_id": None,
        "expected_payment_id": None,
        "expected_settlement_id": None,
        "expected_bank_txn_id": None,
    }

    return data, gt


def scenario_missing_counterparty(case_id: str) -> tuple[dict, dict]:
    data = base_case(case_id, "missing_counterparty")

    data["settlement"] = None

    gt = {
        "case_id": case_id,
        "scenario_type": "missing_counterparty",
        "expected_outcome": "exception",
        "expected_invoice_id": None,
        "expected_payment_id": None,
        "expected_settlement_id": None,
        "expected_bank_txn_id": None,
    }

    return data, gt


def scenario_corrupt_record(case_id: str) -> tuple[dict, dict]:
    data = base_case(case_id, "corrupt_record")

    data["payment"]["amount"] = "NOT_A_NUMBER"
    data["payment"]["utr"] = None

    gt = {
        "case_id": case_id,
        "scenario_type": "corrupt_record",
        "expected_outcome": "exception",
        "expected_invoice_id": None,
        "expected_payment_id": None,
        "expected_settlement_id": None,
        "expected_bank_txn_id": None,
    }

    return data, gt


def scenario_no_valid_match(case_id: str) -> tuple[dict, dict]:
    data = base_case(case_id, "no_valid_match")

    # Deliberately break all meaningful links.
    data["payment"]["invoice_id"] = make_id("INV")
    data["payment"]["utr"] = f"BAD{random.randint(10**10, 10**11 - 1)}"

    data["settlement"]["payment_id"] = make_id("PAY")
    data["settlement"]["utr"] = f"BAD{random.randint(10**10, 10**11 - 1)}"

    data["bank_txn"]["utr"] = f"BAD{random.randint(10**10, 10**11 - 1)}"
    data["bank_txn"]["narration"] = "Unidentified incoming transaction"
    data["bank_txn"]["amount"] = money(
    Decimal(data["bank_txn"]["amount"]) + Decimal("100.00")
)

    gt = {
        "case_id": case_id,
        "scenario_type": "no_valid_match",
        "expected_outcome": "exception",
        "expected_invoice_id": None,
        "expected_payment_id": None,
        "expected_settlement_id": None,
        "expected_bank_txn_id": None,
    }

    return data, gt


def scenario_ambiguous_match(case_id: str) -> tuple[dict, dict]:
    data = base_case(case_id, "ambiguous_match")

    # Keep the invoice -> payment -> settlement chain deterministic.
    # The ambiguity is deliberately isolated to the bank stage: two
    # bank transactions independently satisfy deterministic amount/date
    # checks, while the real transaction has stronger contextual narration.
    real_bank = data["bank_txn"]
    real_bank["narration"] = (
        f"Settlement {data['settlement']['settlement_id']} "
        f"for {data['invoice']['customer_name']}"
    )

    duplicate = dict(real_bank)
    duplicate["bank_txn_id"] = f"BANK_AMBIG_{uuid.uuid4().hex[:8]}"
    duplicate["narration"] = "Incoming transfer"
    data["bank_txn_duplicates"] = [duplicate]

    gt = {
        "case_id": case_id,
        "scenario_type": "ambiguous_match",
        "expected_outcome": "match",
        "expected_invoice_id": data["invoice"]["invoice_id"],
        "expected_payment_id": data["payment"]["payment_id"],
        "expected_settlement_id": data["settlement"]["settlement_id"],
        "expected_bank_txn_id": data["bank_txn"]["bank_txn_id"],
    }

    return data, gt


SCENARIO_BUILDERS = {
    "exact_match": scenario_exact_match,
    "fee_adjusted_match": scenario_fee_adjusted_match,
    "delayed_settlement": scenario_delayed_settlement,
    "fuzzy_reference": scenario_fuzzy_reference,
    "duplicate_candidate": scenario_duplicate_candidate,
    "partial_payment": scenario_partial_payment,
    "missing_counterparty": scenario_missing_counterparty,
    "corrupt_record": scenario_corrupt_record,
    "no_valid_match": scenario_no_valid_match,
    "ambiguous_match": scenario_ambiguous_match,
}


# ============================================================
# GENERATOR
# ============================================================

def generate_cases(seed: int = 42, replace_existing: bool = True) -> list[str]:
    random.seed(seed)
    Faker.seed(seed)

    init_db()

    generated_ids: list[str] = []

    with session_scope() as session:
        if replace_existing:
            session.execute(delete(GroundTruthRow))
            session.execute(delete(Case))

        for scenario_type, count in SCENARIO_MIX.items():
            builder = SCENARIO_BUILDERS[scenario_type]

            for _ in range(count):
                case_id = make_id("CASE")

                payload, gt = builder(case_id)

                # Everything needed by the pipeline is inside payload.
                # Ground truth is stored separately and is never part
                # of the operational reconciliation payload.
                case = Case(
                    case_id=case_id,
                    payload_json=json.dumps(payload, default=str),
                )

                truth = GroundTruthRow(
                    case_id=case_id,
                    scenario_type=gt["scenario_type"],
                    expected_outcome=gt["expected_outcome"],
                    expected_invoice_id=gt["expected_invoice_id"],
                    expected_payment_id=gt["expected_payment_id"],
                    expected_settlement_id=gt["expected_settlement_id"],
                    expected_bank_txn_id=gt["expected_bank_txn_id"],
                )

                session.add(case)
                session.add(truth)

                generated_ids.append(case_id)

    return generated_ids


def get_scenario_counts() -> dict[str, int]:
    counts = {scenario: 0 for scenario in SCENARIO_MIX}

    with session_scope() as session:
        rows = session.query(GroundTruthRow.scenario_type).all()

        for (scenario_type,) in rows:
            if scenario_type in counts:
                counts[scenario_type] += 1

    return counts


if __name__ == "__main__":
    ids = generate_cases()

    print("=" * 60)
    print("ReconPilot Synthetic Dataset")
    print("=" * 60)
    print(f"Generated cases: {len(ids)}")
    print(f"Expected cases:  {TOTAL_CASES}")
    print()
    print("Scenario distribution:")

    counts = get_scenario_counts()

    for scenario, expected in SCENARIO_MIX.items():
        actual = counts.get(scenario, 0)
        status = "OK" if actual == expected else "ERROR"
        print(f"{status:5} {scenario:25} {actual:3} / {expected:3}")

    print()
    print("Dataset generation complete.")