from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional

from rapidfuzz.fuzz import ratio

from reconpilot.schemas import BankTxn, Invoice, Payment, Settlement


MAX_CANDIDATES = 3
DEFAULT_DATE_WINDOW_DAYS = 14
AMOUNT_TOLERANCE = Decimal("0.01")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    entity_type: str
    score: float
    reasons: tuple[str, ...]


def _amount_score(
    expected: Optional[Decimal],
    actual: Optional[Decimal],
) -> float:
    if expected is None or actual is None:
        return 0.0

    difference = abs(expected - actual)

    if difference <= AMOUNT_TOLERANCE:
        return 1.0

    if expected == 0:
        return 0.0

    relative_difference = float(difference / abs(expected))

    if relative_difference <= 0.01:
        return 0.85

    if relative_difference <= 0.05:
        return 0.60

    if relative_difference <= 0.10:
        return 0.30

    return 0.0


def _date_score(
    expected: Optional[datetime],
    actual: Optional[datetime],
) -> float:
    if expected is None or actual is None:
        return 0.0

    days = abs((expected.date() - actual.date()).days)

    if days == 0:
        return 1.0

    if days <= 3:
        return 0.85

    if days <= 7:
        return 0.65

    if days <= DEFAULT_DATE_WINDOW_DAYS:
        return 0.40

    return 0.0


def _text_score(
    expected: Optional[str],
    actual: Optional[str],
) -> float:
    if not expected or not actual:
        return 0.0
    left = " ".join(expected.strip().lower().split())
    right = " ".join(actual.strip().lower().split())
    if left == right:
        return 1.0
    if left in right or right in left:
        return 0.98
    return ratio(left, right) / 100.0


def _candidate_sort_key(candidate: Candidate) -> tuple[float, str]:
    return (-candidate.score, candidate.candidate_id)


def generate_payment_candidates(
    invoice: Invoice,
    payments: list[Payment],
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> list[Candidate]:
    """
    Generate explainable payment candidates for an invoice.

    This function does NOT decide that a payment is correct.
    It only ranks plausible candidates for later investigation.

    At most three candidates are returned.
    """

    if max_candidates <= 0:
        return []

    candidates: list[Candidate] = []

    for payment in payments:
        reasons: list[str] = []
        signals: list[float] = []

        amount_score = _amount_score(
            invoice.amount,
            payment.amount,
        )

        if amount_score > 0:
            signals.append(amount_score)

            if amount_score == 1.0:
                reasons.append("amount_exact")
            elif amount_score >= 0.85:
                reasons.append("amount_near_exact")
            else:
                reasons.append("amount_similar")

        date_score = _date_score(
            invoice.issued_at,
            payment.paid_at,
        )

        if date_score > 0:
            signals.append(date_score)

            if date_score == 1.0:
                reasons.append("date_exact")
            elif date_score >= 0.85:
                reasons.append("date_within_3_days")
            else:
                reasons.append("date_nearby")

        reference_score = _text_score(
            invoice.reference,
            payment.utr,
        )

        if reference_score > 0:
            signals.append(reference_score)

            if reference_score >= 0.90:
                reasons.append("reference_high_similarity")
            elif reference_score >= 0.70:
                reasons.append("reference_partial_similarity")

        invoice_id_score = _text_score(
            invoice.invoice_id,
            payment.invoice_id,
        )

        if invoice_id_score > 0:
            signals.append(invoice_id_score)

            if invoice_id_score == 1.0:
                reasons.append("invoice_id_exact")
            elif invoice_id_score >= 0.70:
                reasons.append("invoice_id_similar")

        # A candidate must have at least one meaningful reconciliation signal.
        # Tiny fuzzy similarities alone are not sufficient evidence.
        meaningful_signal = (
            amount_score >= 0.60
            or date_score >= 0.65
            or reference_score >= 0.70
            or invoice_id_score >= 0.70
        )

        if not meaningful_signal:
            continue

        # Weighted evidence. Amount receives the greatest weight because
        # financial reconciliation should prioritize economic consistency.
        score = (
            amount_score * 0.50
            + date_score * 0.20
            + reference_score * 0.15
            + invoice_id_score * 0.15
        )

        candidates.append(
            Candidate(
                candidate_id=payment.payment_id,
                entity_type="payment",
                score=round(score, 4),
                reasons=tuple(reasons),
            )
        )

    candidates.sort(key=_candidate_sort_key)

    return candidates[:max_candidates]


def generate_settlement_candidates(
    payment: Payment,
    settlements: list[Settlement],
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> list[Candidate]:
    """
    Generate explainable settlement candidates for a payment.
    """

    if max_candidates <= 0:
        return []

    candidates: list[Candidate] = []

    for settlement in settlements:
        reasons: list[str] = []

        amount_score = _amount_score(
            payment.amount,
            settlement.net_amount,
        )

        date_score = _date_score(
            payment.paid_at,
            settlement.settled_at,
        )

        utr_score = _text_score(
            payment.utr,
            settlement.utr,
        )

        payment_id_score = _text_score(
            payment.payment_id,
            settlement.payment_id,
        )

        if (
            amount_score == 0
            and date_score == 0
            and utr_score == 0
            and payment_id_score == 0
        ):
            continue

        if amount_score > 0:
            reasons.append(
                "amount_exact"
                if amount_score == 1.0
                else "amount_similar"
            )

        if date_score > 0:
            reasons.append(
                "date_exact"
                if date_score == 1.0
                else "date_nearby"
            )

        if utr_score >= 0.90:
            reasons.append("utr_high_similarity")
        elif utr_score >= 0.70:
            reasons.append("utr_partial_similarity")

        if payment_id_score == 1.0:
            reasons.append("payment_id_exact")
        elif payment_id_score >= 0.70:
            reasons.append("payment_id_similar")

        score = (
            amount_score * 0.45
            + date_score * 0.20
            + utr_score * 0.20
            + payment_id_score * 0.15
        )

        candidates.append(
            Candidate(
                candidate_id=settlement.settlement_id,
                entity_type="settlement",
                score=round(score, 4),
                reasons=tuple(reasons),
            )
        )

    candidates.sort(key=_candidate_sort_key)

    return candidates[:max_candidates]


def generate_bank_candidates(
    settlement: Settlement,
    bank_transactions: list[BankTxn],
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> list[Candidate]:
    """
    Generate explainable bank transaction candidates for a settlement.
    """

    if max_candidates <= 0:
        return []

    candidates: list[Candidate] = []

    for transaction in bank_transactions:
        reasons: list[str] = []

        amount_score = _amount_score(
            settlement.net_amount,
            transaction.amount,
        )

        date_score = _date_score(
            settlement.settled_at,
            transaction.booked_at,
        )

        utr_score = _text_score(
            settlement.utr,
            transaction.utr,
        )

        narration_score = _text_score(
            settlement.settlement_id,
            transaction.narration,
        )
        meaningful_signal = (
            amount_score >= 0.60
            or utr_score >= 0.70
            or narration_score >= 0.70
        )

        if not meaningful_signal:
            continue

        if amount_score > 0:
            reasons.append(
                "amount_exact"
                if amount_score == 1.0
                else "amount_similar"
            )

        if date_score > 0:
            reasons.append(
                "date_exact"
                if date_score == 1.0
                else "date_nearby"
            )

        if utr_score >= 0.90:
            reasons.append("utr_high_similarity")
        elif utr_score >= 0.70:
            reasons.append("utr_partial_similarity")

        if narration_score >= 0.70:
            reasons.append("narration_similarity")

        score = (
            amount_score * 0.50
            + date_score * 0.20
            + utr_score * 0.20
            + narration_score * 0.10
        )

        candidates.append(
            Candidate(
                candidate_id=transaction.bank_txn_id,
                entity_type="bank_txn",
                score=round(score, 4),
                reasons=tuple(reasons),
            )
        )

    candidates.sort(key=_candidate_sort_key)

    return candidates[:max_candidates]