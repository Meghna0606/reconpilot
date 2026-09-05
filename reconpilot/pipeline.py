from __future__ import annotations
import json, uuid
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from pydantic import ValidationError
from reconpilot.audit import log_audit_event
from reconpilot.candidates import generate_bank_candidates, generate_payment_candidates, generate_settlement_candidates
from reconpilot.config import AI_MODE, LITELLM_MODEL
from reconpilot.db import init_db, session_scope
from reconpilot.decision_gate import apply_decision_gate
from reconpilot.evaluation import EvaluationMetrics, calculate_metrics
from reconpilot.generate import generate_cases
from reconpilot.match_deterministic import deterministic_invoice_payment_matches, deterministic_settlement_bank_matches, match_payment_settlement, match_settlement_bank
from reconpilot.models import Case, CaseResultRow, ExceptionRow, GroundTruthRow, Run
from reconpilot.schemas import BankTxn, Invoice, Payment, Settlement

PAYMENT_SETTLEMENT_WINDOW_DAYS = 30

def _utcnow() -> datetime: return datetime.now(timezone.utc)
def _new_run_id() -> str: return str(uuid.uuid4())

def _parse_payload(payload_json: str) -> tuple[Invoice, Payment, Settlement | None, BankTxn, list[BankTxn]]:
    payload=json.loads(payload_json)
    invoice=Invoice.model_validate(payload["invoice"]); payment=Payment.model_validate(payload["payment"])
    settlement_payload=payload.get("settlement")
    settlement=Settlement.model_validate(settlement_payload) if settlement_payload is not None else None
    bank=BankTxn.model_validate(payload["bank_txn"])
    duplicates=[BankTxn.model_validate(x) for x in payload.get("bank_txn_duplicates", [])]
    return invoice,payment,settlement,bank,[bank,*duplicates]

def _persist_case_result(*, run_id: str, case_id: str, terminal_state: str, reason: str, confidence: float|None=None,
                         matched_invoice_id: str|None=None, matched_payment_id: str|None=None,
                         matched_settlement_id: str|None=None, matched_bank_txn_id: str|None=None,
                         exception_reason: str|None=None, error_type: str|None=None,
                         ai_mode: str|None=None, ai_classification: str|None=None,
                         ai_evidence: list[str]|None=None, ai_uncertainty: list[str]|None=None,
                         ai_selected_candidate_id: str|None=None) -> None:
    with session_scope() as s:
        s.add(CaseResultRow(run_id=run_id,case_id=case_id,terminal_state=terminal_state,reason=reason,confidence=confidence,
            matched_invoice_id=matched_invoice_id,matched_payment_id=matched_payment_id,matched_settlement_id=matched_settlement_id,
            matched_bank_txn_id=matched_bank_txn_id,exception_reason=exception_reason,error_type=error_type,
            ai_mode=ai_mode,ai_classification=ai_classification,
            ai_evidence_json=json.dumps(ai_evidence or []),ai_uncertainty_json=json.dumps(ai_uncertainty or []),
            ai_selected_candidate_id=ai_selected_candidate_id))

def _persist_exception(*, run_id: str, case_id: str, reason_code: str, detail: str, recommended_action: str|None=None) -> None:
    with session_scope() as s:
        s.add(ExceptionRow(run_id=run_id,case_id=case_id,reason_code=reason_code,detail=detail,recommended_action=recommended_action))

def _ground_truth(case_id: str) -> GroundTruthRow|None:
    with session_scope() as s: return s.query(GroundTruthRow).filter(GroundTruthRow.case_id==case_id).first()

def _result_for_evaluation(*, terminal_state: str, ground_truth: GroundTruthRow|None,
                           matched_invoice_id: str|None, matched_payment_id: str|None,
                           matched_settlement_id: str|None, matched_bank_txn_id: str|None,
                           exception_reason: str|None=None) -> dict[str,Any]:
    if terminal_state in {"matched_deterministically","matched_by_ai"}:
        correct=None
        if ground_truth:
            correct=(ground_truth.expected_outcome=="match" and matched_invoice_id==ground_truth.expected_invoice_id and
                     matched_payment_id==ground_truth.expected_payment_id and matched_settlement_id==ground_truth.expected_settlement_id and
                     matched_bank_txn_id==ground_truth.expected_bank_txn_id)
        return {"outcome":"matched","method":"deterministic" if terminal_state=="matched_deterministically" else "ai","correct":correct,"auto_resolved":True}
    if terminal_state=="suggested_match": return {"outcome":"suggested_match","method":"ai","correct":None,"auto_resolved":False}
    return {"outcome":"exception","method":"none","correct":None,"auto_resolved":False,"exception_reason":exception_reason or "unknown"}

def _safe_exception(*, run_id:str, case:Case, gt:GroundTruthRow|None, reason_code:str, reason:str, error_type:str|None=None, recommended_action:str="Manual review") -> dict[str,Any]:
    _persist_case_result(run_id=run_id,case_id=case.case_id,terminal_state="exception",reason=reason,exception_reason=reason_code,error_type=error_type)
    _persist_exception(run_id=run_id,case_id=case.case_id,reason_code=reason_code,detail=reason,recommended_action=recommended_action)
    log_audit_event(run_id=run_id,case_id=case.case_id,stage="recover",action="case_exception",decision="EXCEPTION",reason=reason,error_type=error_type,recovery_action="safe_exception")
    return _result_for_evaluation(terminal_state="exception",ground_truth=gt,matched_invoice_id=None,matched_payment_id=None,matched_settlement_id=None,matched_bank_txn_id=None,exception_reason=reason_code)

def _process_case(*, run_id:str, case:Case, ai_mode:str=AI_MODE, model:str=LITELLM_MODEL) -> dict[str,Any]:
    gt=_ground_truth(case.case_id)
    try:
        invoice,payment,settlement,original_bank,bank_txns=_parse_payload(case.payload_json)
        log_audit_event(run_id=run_id,case_id=case.case_id,stage="load",action="load_case",decision="LOADED",reason="case_payload_loaded")
        if settlement is None: return _safe_exception(run_id=run_id,case=case,gt=gt,reason_code="MISSING_RECORD",reason="missing_settlement_record",recommended_action="Retrieve missing settlement")
        im=deterministic_invoice_payment_matches(invoice,[payment],date_window_days=PAYMENT_SETTLEMENT_WINDOW_DAYS)
        log_audit_event(run_id=run_id,case_id=case.case_id,stage="deterministic",action="invoice_payment_match",decision=im.status.upper(),reason=im.reason,confidence=im.confidence)
        if im.status!="matched":
            c=generate_payment_candidates(invoice,[payment]); log_audit_event(run_id=run_id,case_id=case.case_id,stage="candidates",action="generate_payment_candidates",decision="CANDIDATES_GENERATED",reason=im.reason,candidate_ids=[x.candidate_id for x in c])
            code="INVALID_RECORD" if im.reason.startswith("payment_missing") else "NO_CANDIDATES"
            return _safe_exception(run_id=run_id,case=case,gt=gt,reason_code=code,reason=im.reason,recommended_action="Correct payment source data" if code=="INVALID_RECORD" else "Review payment-to-invoice linkage")
        psm=match_payment_settlement(payment,settlement,date_window_days=PAYMENT_SETTLEMENT_WINDOW_DAYS)
        log_audit_event(run_id=run_id,case_id=case.case_id,stage="deterministic",action="payment_settlement_match",decision=psm.status.upper(),reason=psm.reason,confidence=psm.confidence)
        if psm.status!="matched":
            c=generate_settlement_candidates(payment,[settlement]); log_audit_event(run_id=run_id,case_id=case.case_id,stage="candidates",action="generate_settlement_candidates",decision="CANDIDATES_GENERATED",reason=psm.reason,candidate_ids=[x.candidate_id for x in c])
            return _safe_exception(run_id=run_id,case=case,gt=gt,reason_code="NO_CANDIDATES",reason=psm.reason,recommended_action="Review payment/settlement linkage")
        sbm=deterministic_settlement_bank_matches(settlement,bank_txns)
        log_audit_event(run_id=run_id,case_id=case.case_id,stage="deterministic",action="settlement_bank_match",decision=sbm.status.upper(),reason=sbm.reason,confidence=sbm.confidence)
        if sbm.status=="matched":
            confidence=min(x for x in (im.confidence,psm.confidence,sbm.confidence) if x is not None)
            _persist_case_result(run_id=run_id,case_id=case.case_id,terminal_state="matched_deterministically",reason="all_reconciliation_links_deterministically_validated",confidence=confidence,matched_invoice_id=invoice.invoice_id,matched_payment_id=payment.payment_id,matched_settlement_id=settlement.settlement_id,matched_bank_txn_id=sbm.bank_txn_id)
            log_audit_event(run_id=run_id,case_id=case.case_id,stage="persist",action="persist_match",decision="AUTO_RESOLVE",reason="all_deterministic_checks_passed")
            return _result_for_evaluation(terminal_state="matched_deterministically",ground_truth=gt,matched_invoice_id=invoice.invoice_id,matched_payment_id=payment.payment_id,matched_settlement_id=settlement.settlement_id,matched_bank_txn_id=sbm.bank_txn_id)
        candidates=generate_bank_candidates(settlement,bank_txns)
        log_audit_event(run_id=run_id,case_id=case.case_id,stage="candidates",action="generate_bank_candidates",decision="CANDIDATES_GENERATED",reason="deterministic_bank_match_not_safe",candidate_ids=[x.candidate_id for x in candidates])
        if not candidates: return _safe_exception(run_id=run_id,case=case,gt=gt,reason_code="NO_CANDIDATES",reason=sbm.reason,recommended_action="Review bank transaction")
        from reconpilot.ai_investigator import investigate
        target={"invoice":invoice.model_dump(mode="json"),"payment":payment.model_dump(mode="json"),"settlement":settlement.model_dump(mode="json"),"bank_txn":original_bank.model_dump(mode="json")}
        ai=investigate(target,candidates,mode=ai_mode,model=model)
        if not ai.success or ai.investigation is None:
            failure=ai.failure; et=failure.failure_type if failure else "AI_FAILURE"; detail=failure.detail if failure else "AI investigation failed"
            _persist_case_result(run_id=run_id,case_id=case.case_id,terminal_state="failed_safely",reason=detail,exception_reason=et,error_type=et,ai_mode=ai_mode)
            _persist_exception(run_id=run_id,case_id=case.case_id,reason_code=et,detail=detail,recommended_action="Manual review; configure AI provider" if et not in {"AI_DISABLED"} else "Manual review")
            log_audit_event(run_id=run_id,case_id=case.case_id,stage="recover",action="ai_failure",decision="FAILED_SAFELY",reason=detail,error_type=et,recovery_action="safe_exception")
            return _result_for_evaluation(terminal_state="exception",ground_truth=gt,matched_invoice_id=None,matched_payment_id=None,matched_settlement_id=None,matched_bank_txn_id=None,exception_reason=et)
        inv=ai.investigation
        log_audit_event(run_id=run_id,case_id=case.case_id,stage="llm",action="investigate_case",decision=inv.classification,reason=inv.recommended_action,confidence=inv.confidence,candidate_ids=[inv.selected_candidate_id] if inv.selected_candidate_id else None)
        # IMPORTANT: abstaining classifications do not require a candidate.
        # Match classifications, however, must pass the full defense-in-depth
        # candidate membership and deterministic validation checks below.
        if inv.classification in {"AMBIGUOUS","UNRESOLVABLE","PARTIAL_PAYMENT","DUPLICATE","MISSING_RECORD"} and inv.selected_candidate_id is None:
            decision=apply_decision_gate(inv,deterministic_valid=False)
            log_audit_event(run_id=run_id,case_id=case.case_id,stage="validate",action="apply_decision_gate",decision=decision.decision,reason=decision.reason,confidence=decision.confidence)
            return _safe_exception(run_id=run_id,case=case,gt=gt,reason_code="AMBIGUOUS_MATCH" if inv.classification=="AMBIGUOUS" else "AI_EXCEPTION",reason=decision.reason,recommended_action="Human investigation required")
        selected_id=inv.selected_candidate_id
        candidate_ids={c.candidate_id for c in candidates}
        selected=next((b for b in bank_txns if b.bank_txn_id==selected_id),None)
        if selected_id is None or selected_id not in candidate_ids or selected is None:
            return _safe_exception(run_id=run_id,case=case,gt=gt,reason_code="AI_SAFETY_BLOCK",reason="AI selected a candidate that was not supplied by deterministic candidate generation",recommended_action="Human review")
        selected_match=match_settlement_bank(settlement,selected)
        if selected_match.status!="matched":
            return _safe_exception(run_id=run_id,case=case,gt=gt,reason_code="AI_SAFETY_BLOCK",reason=f"AI-selected candidate failed deterministic validation: {selected_match.reason}",recommended_action="Human review")
        deterministic_valid=im.status=="matched" and psm.status=="matched" and selected_match.status=="matched"
        conflicting=(sbm.status=="matched" and selected_id!=original_bank.bank_txn_id)
        decision=apply_decision_gate(inv,deterministic_valid=deterministic_valid,conflicting_candidate=conflicting)
        log_audit_event(run_id=run_id,case_id=case.case_id,stage="validate",action="apply_decision_gate",decision=decision.decision,reason=decision.reason,confidence=decision.confidence,candidate_ids=[decision.selected_candidate_id] if decision.selected_candidate_id else None)
        if decision.decision=="AUTO_RESOLVE":
            if not deterministic_valid: return _safe_exception(run_id=run_id,case=case,gt=gt,reason_code="AI_SAFETY_BLOCK",reason="final automatic-resolution invariant failed")
            _persist_case_result(run_id=run_id,case_id=case.case_id,terminal_state="matched_by_ai",reason=decision.reason,confidence=decision.confidence,matched_invoice_id=invoice.invoice_id,matched_payment_id=payment.payment_id,matched_settlement_id=settlement.settlement_id,matched_bank_txn_id=selected.bank_txn_id,ai_mode=ai_mode,ai_classification=inv.classification,ai_evidence=inv.evidence,ai_uncertainty=inv.uncertainty,ai_selected_candidate_id=selected.bank_txn_id)
            log_audit_event(run_id=run_id,case_id=case.case_id,stage="persist",action="persist_ai_match",decision="AUTO_RESOLVE",reason=decision.reason,confidence=decision.confidence,candidate_ids=[selected.bank_txn_id])
            return _result_for_evaluation(terminal_state="matched_by_ai",ground_truth=gt,matched_invoice_id=invoice.invoice_id,matched_payment_id=payment.payment_id,matched_settlement_id=settlement.settlement_id,matched_bank_txn_id=selected.bank_txn_id)
        if decision.decision=="SUGGEST_MATCH":
            _persist_case_result(run_id=run_id,case_id=case.case_id,terminal_state="suggested_match",reason=decision.reason,confidence=decision.confidence,ai_mode=ai_mode,ai_classification=inv.classification,ai_evidence=inv.evidence,ai_uncertainty=inv.uncertainty,ai_selected_candidate_id=selected.bank_txn_id)
            _persist_exception(run_id=run_id,case_id=case.case_id,reason_code="AI_SUGGESTION_REQUIRES_REVIEW",detail=decision.reason,recommended_action="Human review")
            return _result_for_evaluation(terminal_state="suggested_match",ground_truth=gt,matched_invoice_id=None,matched_payment_id=None,matched_settlement_id=None,matched_bank_txn_id=None)
        return _safe_exception(run_id=run_id,case=case,gt=gt,reason_code="AI_EXCEPTION",reason=decision.reason,recommended_action="Human investigation required")
    except ValidationError as exc:
        reason=str(exc); _persist_case_result(run_id=run_id,case_id=case.case_id,terminal_state="failed_safely",reason=reason,exception_reason="INVALID_RECORD",error_type="VALIDATION_ERROR")
        _persist_exception(run_id=run_id,case_id=case.case_id,reason_code="INVALID_RECORD",detail=reason,recommended_action="Correct malformed source record")
        log_audit_event(run_id=run_id,case_id=case.case_id,stage="recover",action="validation_failure",decision="FAILED_SAFELY",reason=reason,error_type="VALIDATION_ERROR",recovery_action="safe_exception")
        return {"outcome":"exception","method":"none","correct":None,"auto_resolved":False,"exception_reason":"INVALID_RECORD"}
    except Exception as exc:
        reason=str(exc); _persist_case_result(run_id=run_id,case_id=case.case_id,terminal_state="failed_safely",reason=reason,exception_reason="PIPELINE_FAILURE",error_type=type(exc).__name__)
        _persist_exception(run_id=run_id,case_id=case.case_id,reason_code="PIPELINE_FAILURE",detail=reason,recommended_action="Inspect pipeline logs")
        log_audit_event(run_id=run_id,case_id=case.case_id,stage="recover",action="pipeline_failure",decision="FAILED_SAFELY",reason=reason,error_type=type(exc).__name__,recovery_action="safe_exception")
        return {"outcome":"exception","method":"none","correct":None,"auto_resolved":False,"exception_reason":"PIPELINE_FAILURE"}

def run_reconciliation(*, generate_demo:bool=False, seed:int=42, ai_mode:str|None=None, model:str=LITELLM_MODEL) -> tuple[str,EvaluationMetrics]:
    init_db()
    if generate_demo: generate_cases(seed=seed,replace_existing=True)
    started=perf_counter(); run_id=_new_run_id(); mode=(ai_mode or AI_MODE).lower()
    with session_scope() as s: s.add(Run(run_id=run_id,started_at=_utcnow(),status="running"))
    with session_scope() as s: cases=s.query(Case).order_by(Case.case_id.asc()).all()
    results=[_process_case(run_id=run_id,case=c,ai_mode=mode,model=model) for c in cases]
    elapsed=perf_counter()-started
    total_source_records=0
    for c in cases:
        p=json.loads(c.payload_json); total_source_records += sum(p.get(k) is not None for k in ("invoice","payment","settlement","bank_txn"))
    metrics=calculate_metrics(results,total_source_records=total_source_records,processing_time_seconds=elapsed)
    with session_scope() as s:
        r=s.query(Run).filter(Run.run_id==run_id).first()
        if r: r.finished_at=_utcnow(); r.status="completed"
    return run_id,metrics
