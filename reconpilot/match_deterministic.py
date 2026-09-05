from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from reconpilot.schemas import BankTxn, Invoice, Payment, Settlement


@dataclass(frozen=True)
class MatchResult:
    """
    Result of deterministic reconciliation.

    status:
        matched       -> deterministic evidence is sufficient
        no_match      -> no deterministic candidate
        ambiguous     -> evidence exists but is not safe to auto-match
    """

    status: str
    reason: str

    invoice_id: Optional[str] = None
    payment_id: Optional[str] = None
    settlement_id: Optional[str] = None
    bank_txn_id: Optional[str] = None

    confidence: Optional[float] = None


DEFAULT_DATE_WINDOW_DAYS = 3
FEE_TOLERANCE = Decimal("0.01")


def _days_between(a: Optional[datetime], b: Optional[datetime]) -> Optional[int]:
    if a is None or b is None:
        return None

    return abs((a.date() - b.date()).days)


def _amount_equal(a: Optional[Decimal], b: Optional[Decimal]) -> bool:
    if a is None or b is None:
        return False

    return abs(a - b) <= FEE_TOLERANCE


def _valid_date_pair(
    a: Optional[datetime],
    b: Optional[datetime],
    window_days: int = DEFAULT_DATE_WINDOW_DAYS,
) -> bool:
    delta = _days_between(a, b)

    if delta is None:
        return False

    return delta <= window_days


def _result(
    status: str,
    reason: str,
    *,
    invoice: Optional[Invoice] = None,
    payment: Optional[Payment] = None,
    settlement: Optional[Settlement] = None,
    bank_txn: Optional[BankTxn] = None,
    confidence: Optional[float] = None,
) -> MatchResult:
    return MatchResult(
        status=status,
        reason=reason,
        invoice_id=invoice.invoice_id if invoice else None,
        payment_id=payment.payment_id if payment else None,
        settlement_id=settlement.settlement_id if settlement else None,
        bank_txn_id=bank_txn.bank_txn_id if bank_txn else None,
        confidence=confidence,
    )


def match_invoice_payment(
    invoice: Invoice,
    payment: Payment,
    *,
    date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
) -> MatchResult:
    """
    Safely reconcile one invoice against one payment.

    Deterministic rules:
    1. Exact invoice ID reference + amount + date.
    2. Exact UTR is strong evidence, but amount/date still must agree.
    3. Exact amount + tight date window.
    """

    # Missing critical payment data means we cannot prove a match.
    if payment.amount is None or payment.paid_at is None:
        return _result(
            "no_match",
            "payment_missing_amount_or_date",
        )

    # Strongest path: invoice ID explicitly references this invoice.
    if payment.invoice_id == invoice.invoice_id:
        if not _amount_equal(payment.amount, invoice.amount):
            return _result(
                "ambiguous",
                "invoice_id_matches_but_amount_conflicts",
            )

        if not _valid_date_pair(
            invoice.issued_at,
            payment.paid_at,
            date_window_days,
        ):
            return _result(
                "ambiguous",
                "invoice_id_matches_but_date_is_outside_window",
            )

        return _result(
            "matched",
            "exact_invoice_id_amount_and_date",
            invoice=invoice,
            payment=payment,
            confidence=1.0,
        )

    # Strong UTR evidence is useful only when the economic values agree.
    if payment.utr:
        # A UTR by itself cannot prove which invoice this payment belongs to.
        # Without an invoice reference, require amount + date evidence.
        if _amount_equal(payment.amount, invoice.amount) and _valid_date_pair(
            invoice.issued_at,
            payment.paid_at,
            date_window_days,
        ):
            return _result(
                "matched",
                "utr_amount_and_date_agree",
                invoice=invoice,
                payment=payment,
                confidence=0.98,
            )

    # Weak deterministic path: amount + date.
    if _amount_equal(payment.amount, invoice.amount):
        if _valid_date_pair(
            invoice.issued_at,
            payment.paid_at,
            date_window_days,
        ):
            return _result(
                "matched",
                "exact_amount_and_date_window",
                invoice=invoice,
                payment=payment,
                confidence=0.95,
            )

        return _result(
            "ambiguous",
            "amount_matches_but_date_is_outside_window",
        )

    return _result(
        "no_match",
        "invoice_and_payment_amounts_do_not_match",
    )


def match_payment_settlement(
    payment: Payment,
    settlement: Settlement,
    *,
    date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
) -> MatchResult:
    """
    Reconcile payment -> settlement.

    Supports:
    - exact payment ID
    - exact UTR
    - gross/net fee-aware settlement
    """

    if settlement.net_amount is None:
        return _result(
            "no_match",
            "settlement_missing_net_amount",
        )

    if payment.amount is None:
        return _result(
            "no_match",
            "payment_missing_amount",
        )

    if settlement.payment_id == payment.payment_id:
        if _amount_equal(settlement.net_amount, payment.amount):
            if settlement.settled_at is None or payment.paid_at is None:
                return _result(
                    "matched",
                    "exact_payment_id_and_amount",
                    payment=payment,
                    settlement=settlement,
                    confidence=1.0,
                )

            if _valid_date_pair(
                payment.paid_at,
                settlement.settled_at,
                date_window_days,
            ):
                return _result(
                    "matched",
                    "exact_payment_id_amount_and_date",
                    payment=payment,
                    settlement=settlement,
                    confidence=1.0,
                )

            return _result(
                "ambiguous",
                "payment_id_matches_but_settlement_date_is_outside_window",
            )

        # Payment amount may differ from settlement because of fees.
        if (
            settlement.gross_amount is not None
            and _amount_equal(settlement.gross_amount, payment.amount)
            and settlement.fee_amount is not None
            and _amount_equal(
                settlement.gross_amount - settlement.fee_amount,
                settlement.net_amount,
            )
        ):
            return _result(
                "matched",
                "payment_id_and_fee_adjusted_settlement",
                payment=payment,
                settlement=settlement,
                confidence=0.99,
            )

        return _result(
            "ambiguous",
            "payment_id_matches_but_amount_conflicts",
        )

    # UTR matching.
    if payment.utr and settlement.utr and payment.utr == settlement.utr:
        if settlement.gross_amount is not None and _amount_equal(
            settlement.gross_amount,
            payment.amount,
        ):
            return _result(
                "matched",
                "exact_utr_and_gross_amount",
                payment=payment,
                settlement=settlement,
                confidence=0.99,
            )

    return _result(
        "no_match",
        "no_deterministic_payment_settlement_match",
    )


def match_settlement_bank(
    settlement: Settlement,
    bank_txn: BankTxn,
    *,
    date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
) -> MatchResult:
    """
    Reconcile settlement -> bank transaction.

    Requires strong amount/date/UTR evidence.
    """

    if settlement.net_amount is None or bank_txn.amount is None:
        return _result(
            "no_match",
            "missing_settlement_or_bank_amount",
        )

    # Exact UTR + amount is strongest.
    if settlement.utr and bank_txn.utr and settlement.utr == bank_txn.utr:
        if _amount_equal(settlement.net_amount, bank_txn.amount):
            return _result(
                "matched",
                "exact_utr_and_amount",
                settlement=settlement,
                bank_txn=bank_txn,
                confidence=1.0,
            )

    # Amount + date is acceptable when there is no conflicting evidence.
    if _amount_equal(settlement.net_amount, bank_txn.amount):
        if settlement.settled_at is None or bank_txn.booked_at is None:
            return _result(
                "ambiguous",
                "amount_matches_but_date_is_missing",
            )

        if _valid_date_pair(
            settlement.settled_at,
            bank_txn.booked_at,
            date_window_days,
        ):
            return _result(
                "matched",
                "exact_amount_and_date_window",
                settlement=settlement,
                bank_txn=bank_txn,
                confidence=0.95,
            )

        return _result(
            "ambiguous",
            "amount_matches_but_date_is_outside_window",
        )

    return _result(
        "no_match",
        "settlement_and_bank_amounts_do_not_match",
    )


def deterministic_invoice_payment_matches(
    invoice: Invoice,
    payments: list[Payment],
    *,
    date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
) -> MatchResult:
    """
    Evaluate all payments for an invoice.

    Critical safety rule:
    Never auto-match if more than one payment independently satisfies
    deterministic rules.
    """

    matches: list[MatchResult] = []

    for payment in payments:
        result = match_invoice_payment(
            invoice,
            payment,
            date_window_days=date_window_days,
        )

        if result.status == "matched":
            matches.append(result)

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        return _result(
            "ambiguous",
            "multiple_deterministic_candidates",
        )

    return _result(
        "no_match",
        "no_unique_deterministic_payment_match",
    )

def deterministic_settlement_bank_matches(
    settlement: Settlement,
    bank_transactions: list[BankTxn],
    *,
    date_window_days: int = DEFAULT_DATE_WINDOW_DAYS,
) -> MatchResult:
    """Evaluate all bank transactions and reject non-unique matches.

    A single deterministic bank match is safe. Multiple independently
    valid bank matches are ambiguous and must be escalated rather than
    silently choosing the first row.
    """
    matches: list[MatchResult] = []
    for bank_txn in bank_transactions:
        result = match_settlement_bank(
            settlement, bank_txn, date_window_days=date_window_days
        )
        if result.status == "matched":
            matches.append(result)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return _result(
            "ambiguous",
            "multiple_deterministic_bank_candidates",
        )
    return _result(
        "no_match",
        "no_unique_deterministic_settlement_bank_match",
    )
