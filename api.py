from __future__ import annotations
import json
from fastapi import FastAPI, HTTPException
from reconpilot.audit import get_case_audit, get_run_audit
from reconpilot.db import init_db, session_scope
from reconpilot.models import Case, CaseResultRow, ExceptionRow, GroundTruthRow, Run
from reconpilot.pipeline import run_reconciliation

app=FastAPI(title="ReconPilot API",version="1.0.0",description="Deterministic-first, auditable financial reconciliation controller")
init_db()

@app.get("/health")
def health(): return {"status":"ok","service":"reconpilot"}

@app.get("/runs")
def runs(limit:int=20):
    with session_scope() as s:
        rows=s.query(Run).order_by(Run.started_at.desc()).limit(min(limit,100)).all()
        return [{"run_id":r.run_id,"started_at":r.started_at,"finished_at":r.finished_at,"status":r.status} for r in rows]

@app.post("/runs")
def create_run(seed:int=42,mode:str="local"):
    if mode not in {"local","live","off"}: raise HTTPException(400,"mode must be local, live, or off")
    run_id,metrics=run_reconciliation(generate_demo=True,seed=seed,ai_mode=mode)
    return {"run_id":run_id,"metrics":metrics.__dict__}

def _get_run(run_id):
    with session_scope() as s:
        if not s.query(Run).filter(Run.run_id==run_id).first(): raise HTTPException(404,"run not found")

def _metrics(run_id):
    with session_scope() as s:
        rows=s.query(CaseResultRow).filter(CaseResultRow.run_id==run_id).all()
    total=len(rows); resolved=sum(r.terminal_state in {"matched_deterministically","matched_by_ai"} for r in rows); det=sum(r.terminal_state=="matched_deterministically" for r in rows); ai=sum(r.terminal_state=="matched_by_ai" for r in rows); sugg=sum(r.terminal_state=="suggested_match" for r in rows); exc=total-resolved-sugg
    correct=0
    with session_scope() as s:
        truths={x.case_id:x for x in s.query(GroundTruthRow).all()}
    for r in rows:
        g=truths.get(r.case_id); correct += int(r.terminal_state in {"matched_deterministically","matched_by_ai"} and g and g.expected_outcome=="match" and r.matched_invoice_id==g.expected_invoice_id and r.matched_payment_id==g.expected_payment_id and r.matched_settlement_id==g.expected_settlement_id and r.matched_bank_txn_id==g.expected_bank_txn_id)
    false_auto=resolved-correct
    return {"total_cases":total,"resolved":resolved,"deterministic_resolutions":det,"ai_assisted_resolutions":ai,"suggested_matches":sugg,"exceptions":exc,"resolution_rate":resolved/total if total else 0,"resolved_accuracy":correct/resolved if resolved else 0,"false_auto_matches":false_auto}

@app.get("/runs/{run_id}")
def run_detail(run_id:str):
    _get_run(run_id)
    return {"run_id":run_id,"metrics":_metrics(run_id)}

@app.get("/runs/{run_id}/metrics")
def metrics(run_id:str): _get_run(run_id); return _metrics(run_id)

@app.get("/runs/{run_id}/exceptions")
def exceptions(run_id:str):
    _get_run(run_id)
    with session_scope() as s:
        rows=s.query(ExceptionRow).filter(ExceptionRow.run_id==run_id).order_by(ExceptionRow.id).all()
        return [{"case_id":r.case_id,"reason_code":r.reason_code,"detail":r.detail,"recommended_action":r.recommended_action} for r in rows]

@app.get("/runs/{run_id}/audit")
def audit(run_id:str): _get_run(run_id); return [{"case_id":a.case_id,"stage":a.stage,"action":a.action,"decision":a.decision,"reason":a.reason,"confidence":a.confidence,"candidates":json.loads(a.candidate_ids_json) if a.candidate_ids_json else []} for a in get_run_audit(run_id)]

@app.get("/runs/{run_id}/cases/{case_id}")
def case_detail(run_id:str,case_id:str):
    _get_run(run_id)
    with session_scope() as s:
        result=s.query(CaseResultRow).filter(CaseResultRow.run_id==run_id,CaseResultRow.case_id==case_id).first(); case=s.query(Case).filter(Case.case_id==case_id).first(); gt=s.query(GroundTruthRow).filter(GroundTruthRow.case_id==case_id).first()
    if not result or not case: raise HTTPException(404,"case not found")
    return {"case_id":case_id,"payload":json.loads(case.payload_json),"result":{k:v for k,v in result.__dict__.items() if not k.startswith("_")},"ground_truth":{k:v for k,v in gt.__dict__.items() if not k.startswith("_")} if gt else None,"audit":[{k:v for k,v in a.__dict__.items() if not k.startswith("_")} for a in get_case_audit(case_id) if a.run_id==run_id]}
