from datetime import datetime, timedelta
from decimal import Decimal

from reconpilot.match_deterministic import (
    match_invoice_payment,
    match_payment_settlement,
    match_settlement_bank,
    deterministic_invoice_payment_matches,
)
from reconpilot.schemas import Invoice, Payment, Settlement, BankTxn


def make_invoice(
    amount="1000.00",
    invoice_id="INV-001",
):
    return Invoice(
        invoice_id=invoice_id,
        customer_name="Test Customer",
        amount=Decimal(amount),
        issued_at=datetime(2026, 8, 1),
    )


def make_payment(
    payment_id="PAY-001",
    invoice_id=None,
    amount="1000.00",
    paid_at=datetime(2026, 8, 2),
    utr=None,
):
    return Payment(
        payment_id=payment_id,
        invoice_id=invoice_id,
        amount=Decimal(amount) if amount is not None else None,
        paid_at=paid_at,
        utr=utr,
    )


def test_exact_invoice_id_match():
    invoice = make_invoice()

    payment = make_payment(
        invoice_id="INV-001",
    )

    result = match_invoice_payment(invoice, payment)

    assert result.status == "matched"
    assert result.invoice_id == "INV-001"
    assert result.payment_id == "PAY-001"
    assert result.confidence == 1.0


def test_utr_amount_and_date_match():
    invoice = make_invoice()

    payment = make_payment(
        invoice_id=None,
        utr="UTR-123",
    )

    result = match_invoice_payment(invoice, payment)

    assert result.status == "matched"
    assert result.reason == "utr_amount_and_date_agree"


def test_amount_match_outside_date_window_is_not_auto_matched():
    invoice = make_invoice()

    payment = make_payment(
        paid_at=datetime(2026, 8, 20),
    )

    result = match_invoice_payment(
        invoice,
        payment,
        date_window_days=3,
    )

    assert result.status == "ambiguous"
    assert "date" in result.reason


def test_fee_adjusted_settlement_match():
    payment = make_payment(
        amount="1000.00",
    )

    settlement = Settlement(
        settlement_id="SET-001",
        payment_id="PAY-001",
        gross_amount=Decimal("1000.00"),
        fee_amount=Decimal("20.00"),
        net_amount=Decimal("980.00"),
        settled_at=datetime(2026, 8, 3),
    )

    # The current matcher intentionally treats the payment amount as the
    # economic gross amount for fee-aware reconciliation.
    result = match_payment_settlement(payment, settlement)

    assert result.status == "matched"
    assert result.reason == "payment_id_and_fee_adjusted_settlement"


def test_bank_settlement_utr_match():
    settlement = Settlement(
        settlement_id="SET-001",
        net_amount=Decimal("980.00"),
        utr="SET-UTR-1",
        settled_at=datetime(2026, 8, 3),
    )

    bank = BankTxn(
        bank_txn_id="BANK-001",
        amount=Decimal("980.00"),
        booked_at=datetime(2026, 8, 4),
        utr="SET-UTR-1",
    )

    result = match_settlement_bank(settlement, bank)

    assert result.status == "matched"
    assert result.bank_txn_id == "BANK-001"


def test_duplicate_candidates_are_rejected():
    invoice = make_invoice()

    payment1 = make_payment(
        payment_id="PAY-001",
        invoice_id=None,
    )

    payment2 = make_payment(
        payment_id="PAY-002",
        invoice_id=None,
    )

    result = deterministic_invoice_payment_matches(
        invoice,
        [payment1, payment2],
    )

    assert result.status == "ambiguous"
    assert result.reason == "multiple_deterministic_candidates"


def test_partial_payment_is_not_silently_matched():
    invoice = make_invoice(amount="1000.00")

    payment = make_payment(
        amount="500.00",
        invoice_id="INV-001",
    )

    result = match_invoice_payment(invoice, payment)

    assert result.status == "ambiguous"
    assert result.reason == "invoice_id_matches_but_amount_conflicts"


def test_missing_payment_amount_is_rejected():
    invoice = make_invoice()

    payment = make_payment(
        amount=None,
    )

    result = match_invoice_payment(invoice, payment)

    assert result.status == "no_match"
    assert result.reason == "payment_missing_amount_or_date"

def test_multiple_valid_bank_matches_are_ambiguous():
    from reconpilot.match_deterministic import deterministic_settlement_bank_matches

    settlement = Settlement(
        settlement_id="SET-AMB",
        net_amount=Decimal("1000.00"),
        settled_at=datetime(2026, 8, 3),
        utr="SET-UTR-1",
    )
    banks = [
        BankTxn(bank_txn_id="BANK-001", amount=Decimal("1000.00"),
                booked_at=datetime(2026, 8, 4), utr="SET-UTR-1"),
        BankTxn(bank_txn_id="BANK-002", amount=Decimal("1000.00"),
                booked_at=datetime(2026, 8, 4), utr="SET-UTR-1"),
    ]

    result = deterministic_settlement_bank_matches(settlement, banks)

    assert result.status == "ambiguous"
    assert result.reason == "multiple_deterministic_bank_candidates"
