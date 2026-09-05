from datetime import datetime
from decimal import Decimal

from reconpilot.candidates import (
    generate_bank_candidates,
    generate_payment_candidates,
    generate_settlement_candidates,
)
from reconpilot.schemas import BankTxn, Invoice, Payment, Settlement


def make_invoice():
    return Invoice(
        invoice_id="INV-001",
        customer_name="Test Customer",
        amount=Decimal("1000.00"),
        issued_at=datetime(2026, 8, 1),
        reference="UTR-ABC-123",
    )


def make_payment(
    payment_id,
    amount="1000.00",
    paid_at=datetime(2026, 8, 2),
    utr=None,
):
    return Payment(
        payment_id=payment_id,
        amount=Decimal(amount),
        paid_at=paid_at,
        utr=utr,
    )


def test_payment_candidates_are_limited_to_three():
    invoice = make_invoice()

    payments = [
        make_payment(f"PAY-{i}")
        for i in range(10)
    ]

    candidates = generate_payment_candidates(
        invoice,
        payments,
    )

    assert len(candidates) == 3


def test_payment_candidates_are_ranked_by_evidence():
    invoice = make_invoice()

    strong = make_payment(
        "PAY-STRONG",
        utr="UTR-ABC-123",
    )

    weak = make_payment(
        "PAY-WEAK",
        amount="700.00",
        paid_at=datetime(2026, 8, 20),
    )

    candidates = generate_payment_candidates(
        invoice,
        [weak, strong],
    )

    assert candidates
    assert candidates[0].candidate_id == "PAY-STRONG"


def test_candidate_contains_explainable_reasons():
    invoice = make_invoice()

    payment = make_payment(
        "PAY-001",
        utr="UTR-ABC-123",
    )

    candidates = generate_payment_candidates(
        invoice,
        [payment],
    )

    assert len(candidates) == 1
    assert candidates[0].reasons
    assert "amount_exact" in candidates[0].reasons


def test_unrelated_payment_is_not_candidate():
    invoice = make_invoice()

    payment = make_payment(
        "PAY-001",
        amount="5000.00",
        paid_at=datetime(2026, 10, 30),
        utr="COMPLETELY-DIFFERENT",
    )

    candidates = generate_payment_candidates(
        invoice,
        [payment],
    )

    assert candidates == []


def test_settlement_candidates_are_limited_to_three():
    payment = make_payment("PAY-001")

    settlements = [
        Settlement(
            settlement_id=f"SET-{i}",
            net_amount=Decimal("1000.00"),
            settled_at=datetime(2026, 8, 3),
        )
        for i in range(8)
    ]

    candidates = generate_settlement_candidates(
        payment,
        settlements,
    )

    assert len(candidates) == 3


def test_bank_candidates_are_limited_to_three():
    settlement = Settlement(
        settlement_id="SET-001",
        net_amount=Decimal("1000.00"),
        settled_at=datetime(2026, 8, 3),
    )

    transactions = [
        BankTxn(
            bank_txn_id=f"BANK-{i}",
            amount=Decimal("1000.00"),
            booked_at=datetime(2026, 8, 4),
        )
        for i in range(7)
    ]

    candidates = generate_bank_candidates(
        settlement,
        transactions,
    )

    assert len(candidates) == 3


def test_zero_max_candidates_returns_empty():
    invoice = make_invoice()

    payment = make_payment("PAY-001")

    assert generate_payment_candidates(
        invoice,
        [payment],
        max_candidates=0,
    ) == []