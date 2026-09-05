from __future__ import annotations
import json
import pandas as pd
import streamlit as st
import altair as alt
from reconpilot.audit import get_run_audit
from reconpilot.db import session_scope, init_db
from reconpilot.generate import SCENARIO_MIX, generate_cases
from reconpilot.models import Case, CaseResultRow, ExceptionRow, GroundTruthRow, Run
from reconpilot.pipeline import run_reconciliation
from reconpilot.config import AI_MODE

init_db(); st.set_page_config(page_title="ReconPilot — AI Finance Controller",page_icon="💰",layout="wide")
st.title("ReconPilot")
st.caption("AI Finance Controller • deterministic-first reconciliation • fail-closed automation • auditable decisions")
if "last_run_id" not in st.session_state: st.session_state.last_run_id=None
with st.sidebar:
    st.header("Demo controls")
    seed=st.number_input("Benchmark seed",min_value=0,value=42,step=1)
    mode=st.selectbox("AI execution mode",["local","live","off"],index=["local","live","off"].index(AI_MODE) if AI_MODE in {"local","live","off"} else 0)
    st.caption({"local":"LOCAL_EVALUATOR — reproducible, not an LLM","live":"REAL_LLM — uses LiteLLM/provider credentials","off":"AI disabled — ambiguous cases abstain"}[mode])
    if st.button("Generate 104-case batch",use_container_width=True):
        ids=generate_cases(seed=int(seed),replace_existing=True); st.session_state.last_run_id=None; st.success(f"Generated {len(ids)} cases")
    if st.button("Run ReconPilot",type="primary",use_container_width=True):
        rid, _ = run_reconciliation(
            generate_demo=True,
            seed=int(seed),
            ai_mode=mode,
        )
        st.session_state.last_run_id = rid
        st.success(f"✓ Completed • Run ID: {rid[:8]}…")
        st.caption("Batch: 104 cases processed")
    st.divider(); st.caption("Safety invariant: AI confidence never independently creates an auto-match.")
run_id=st.session_state.last_run_id
if not run_id:
    st.info("Generate and run the benchmark from the sidebar.")
    st.subheader("Benchmark scenario mix")
    st.dataframe(pd.DataFrame([{"Scenario":k,"Cases":v} for k,v in SCENARIO_MIX.items()]),use_container_width=True,hide_index=True)
    st.stop()
with session_scope() as s:
    run=s.query(Run).filter(Run.run_id==run_id).first(); rows=s.query(CaseResultRow).filter(CaseResultRow.run_id==run_id).all(); exceptions=s.query(ExceptionRow).filter(ExceptionRow.run_id==run_id).all(); truths={x.case_id:x for x in s.query(GroundTruthRow).all()}; cases={x.case_id:x for x in s.query(Case).all()}
resolved=[r for r in rows if r.terminal_state in {"matched_deterministically","matched_by_ai"}]; det=[r for r in resolved if r.terminal_state=="matched_deterministically"]; ai=[r for r in resolved if r.terminal_state=="matched_by_ai"]; suggested=[r for r in rows if r.terminal_state=="suggested_match"]
correct=[]
for r in resolved:
    g=truths.get(r.case_id); correct.append(bool(g and g.expected_outcome=="match" and r.matched_invoice_id==g.expected_invoice_id and r.matched_payment_id==g.expected_payment_id and r.matched_settlement_id==g.expected_settlement_id and r.matched_bank_txn_id==g.expected_bank_txn_id))
false_auto=len(resolved)-sum(correct)
cols=st.columns(6); cols[0].metric("Cases",len(rows)); cols[1].metric("Resolved",len(resolved)); cols[2].metric("Resolution",f"{len(resolved)/len(rows):.1%}" if rows else "0%"); cols[3].metric("Resolved accuracy",f"{sum(correct)/len(correct):.1%}" if correct else "0%"); cols[4].metric("AI-assisted",len(ai)); cols[5].metric("False auto",false_auto)
st.subheader("Controller decision mix")

decision_df = pd.DataFrame({
    "Decision": [
        "Deterministic auto",
        "AI auto",
        "Suggested",
        "Exception",
    ],
    "Cases": [
        len(det),
        len(ai),
        len(suggested),
        len(rows) - len(resolved) - len(suggested),
    ],
})

decision_chart = (
    alt.Chart(decision_df)
    .mark_bar()
    .encode(
        x=alt.X(
            "Decision:N",
            sort=None,
            axis=alt.Axis(
                title=None,
                labelAngle=0,
                labelLimit=180,
            ),
        ),
        y=alt.Y(
            "Cases:Q",
            title="Cases",
            axis=alt.Axis(
                format="d",
                labelPadding=8,
                titlePadding=10,
            ),
            scale=alt.Scale(zero=True),
        ),
        tooltip=[
            alt.Tooltip("Decision:N", title="Decision"),
            alt.Tooltip("Cases:Q", title="Cases", format="d"),
        ],
    )
    .properties(height=400)
    .configure_view(stroke=None)
    .configure_axis(
        labelFontSize=13,
        titleFontSize=13,
    )
)

st.altair_chart(decision_chart, use_container_width=True)
st.subheader("Exception queue")
st.dataframe(pd.DataFrame([{"Case":e.case_id,"Type":e.reason_code,"Detail":e.detail,"Recommended action":e.recommended_action} for e in exceptions]),use_container_width=True,hide_index=True)
st.subheader("Scenario performance")
scenario_rows=[]
for scenario in SCENARIO_MIX:
    subset=[r for r in rows if truths.get(r.case_id) and truths[r.case_id].scenario_type==scenario]; rr=sum(r.terminal_state in {"matched_deterministically","matched_by_ai"} for r in subset); cc=sum(bool((lambda g,r:g and g.expected_outcome=="match" and r.matched_invoice_id==g.expected_invoice_id and r.matched_payment_id==g.expected_payment_id and r.matched_settlement_id==g.expected_settlement_id and r.matched_bank_txn_id==g.expected_bank_txn_id)(truths.get(r.case_id),r)) for r in subset)
    scenario_rows.append({"Scenario":scenario,"Cases":len(subset),"Resolved":rr,"Correct":cc,"Exceptions":len(subset)-rr,"Resolution":rr/len(subset) if subset else 0})
st.dataframe(pd.DataFrame(scenario_rows),use_container_width=True,hide_index=True)
st.subheader("Case investigation")
case_ids=[r.case_id for r in rows]; selected=st.selectbox("Select case",case_ids)
r=next(r for r in rows if r.case_id==selected); g=truths.get(selected); payload=json.loads(cases[selected].payload_json) if selected in cases else {}
c1,c2=st.columns(2)
with c1: st.markdown("**Decision**"); st.json({"terminal_state":r.terminal_state,"confidence":r.confidence,"reason":r.reason,"matched_bank_txn":r.matched_bank_txn_id,"exception":r.exception_reason})
with c2:
    st.markdown("**AI evidence**")
    st.json({"mode":r.ai_mode,"classification":r.ai_classification,"selected_candidate":r.ai_selected_candidate_id,"evidence":json.loads(r.ai_evidence_json or "[]"),"uncertainty":json.loads(r.ai_uncertainty_json or "[]")})
st.markdown("**Reconciliation chain**"); st.write(f"Invoice `{payload.get('invoice',{}).get('invoice_id')}` → Payment `{payload.get('payment',{}).get('payment_id')}` → Settlement `{payload.get('settlement',{}).get('settlement_id') if payload.get('settlement') else 'MISSING'}` → Bank `{r.matched_bank_txn_id or 'REVIEW'}`")
st.subheader("Audit trail")
aud=get_run_audit(run_id); st.dataframe(pd.DataFrame([{"Time":a.timestamp,"Stage":a.stage,"Action":a.action,"Decision":a.decision,"Reason":a.reason,"Confidence":a.confidence,"Candidates":a.candidate_ids_json} for a in aud if a.case_id==selected]),use_container_width=True,hide_index=True)
