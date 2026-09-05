from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any
from reconpilot.config import BENCHMARK_DIR
from reconpilot.db import init_db, session_scope
from reconpilot.generate import SCENARIO_MIX, generate_cases
from reconpilot.match_deterministic import deterministic_invoice_payment_matches, deterministic_settlement_bank_matches, match_payment_settlement
from reconpilot.candidates import generate_bank_candidates
from reconpilot.models import Case, CaseResultRow, ExceptionRow, GroundTruthRow
from reconpilot.pipeline import _parse_payload, run_reconciliation


def _baseline_case(case: Case) -> dict[str, Any]:
    try:
        invoice,payment,settlement,bank,banks=_parse_payload(case.payload_json)
        if settlement is None: return {"outcome":"exception","reason":"MISSING_RECORD"}
        im=deterministic_invoice_payment_matches(invoice,[payment],date_window_days=3)
        if im.status!="matched": return {"outcome":"exception","reason":"NO_CANDIDATES"}
        psm=match_payment_settlement(payment,settlement,date_window_days=3)
        if psm.status!="matched": return {"outcome":"exception","reason":"NO_CANDIDATES"}
        sb=deterministic_settlement_bank_matches(settlement,banks)
        if sb.status!="matched": return {"outcome":"exception","reason":"AMBIGUOUS_MATCH" if sb.status=="ambiguous" else "NO_CANDIDATES"}
        return {"outcome":"matched","method":"baseline","bank_txn_id":sb.bank_txn_id}
    except Exception:
        return {"outcome":"exception","reason":"INVALID_RECORD"}


def baseline_report() -> dict[str, Any]:
    with session_scope() as s:
        cases=s.query(Case).order_by(Case.case_id).all()
        truths={x.case_id:x for x in s.query(GroundTruthRow).all()}
    start=perf_counter(); rows=[]
    for c in cases:
        r=_baseline_case(c); gt=truths.get(c.case_id)
        correct=bool(r["outcome"]=="matched" and gt and gt.expected_outcome=="match" and r.get("bank_txn_id")==gt.expected_bank_txn_id)
        rows.append({**r,"case_id":c.case_id,"scenario":gt.scenario_type if gt else "unknown","correct":correct})
    elapsed=perf_counter()-start
    matched=sum(r["outcome"]=="matched" for r in rows); correct=sum(r["correct"] for r in rows)
    return {"name":"deterministic_baseline","total_cases":len(rows),"matched":matched,"exceptions":len(rows)-matched,
            "resolution_rate":matched/len(rows) if rows else 0,"resolved_accuracy":correct/matched if matched else 0,
            "overall_accuracy":correct/len(rows) if rows else 0,"false_auto_matches":sum(r["outcome"]=="matched" and not r["correct"] for r in rows),
            "throughput_cases_per_second":len(rows)/elapsed if elapsed else 0,"processing_time_seconds":elapsed,
            "scenario_metrics":_scenario_rows(rows)}


def _scenario_rows(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    out=[]
    for scenario in SCENARIO_MIX:
        subset=[r for r in rows if r.get("scenario")==scenario]
        matched=sum(r.get("outcome")=="matched" for r in subset); correct=sum(bool(r.get("correct")) for r in subset)
        out.append({"scenario":scenario,"cases":len(subset),"resolved":matched,"correct_resolved":correct,
                    "exceptions":len(subset)-matched,"resolution_rate":matched/len(subset) if subset else 0,
                    "resolved_accuracy":correct/matched if matched else 0})
    return out


def _candidate_quality() -> dict[str,float]:
    with session_scope() as s:
        cases=s.query(Case).all(); truths={x.case_id:x for x in s.query(GroundTruthRow).all()}
    eligible=hit1=hit3=0
    for case in cases:
        gt=truths.get(case.case_id)
        if not gt or gt.expected_outcome!="match" or not gt.expected_bank_txn_id: continue
        try:
            _,_,settlement,_,banks=_parse_payload(case.payload_json)
            if settlement is None: continue
            ids=[c.candidate_id for c in generate_bank_candidates(settlement,banks,max_candidates=3)]
        except Exception: continue
        eligible+=1; hit1+=int(bool(ids and ids[0]==gt.expected_bank_txn_id)); hit3+=int(gt.expected_bank_txn_id in ids[:3])
    return {"eligible_match_cases":eligible,"recall_at_1":hit1/eligible if eligible else 0,"recall_at_3":hit3/eligible if eligible else 0}


def _controller_report(run_id:str, metrics:Any) -> dict[str,Any]:
    with session_scope() as s:
        results=s.query(CaseResultRow).filter(CaseResultRow.run_id==run_id).all()
        truths={x.case_id:x for x in s.query(GroundTruthRow).all()}
        exceptions=s.query(ExceptionRow).filter(ExceptionRow.run_id==run_id).all()
    rows=[]
    for r in results:
        gt=truths.get(r.case_id); correct=bool(r.terminal_state in {"matched_deterministically","matched_by_ai"} and gt and gt.expected_outcome=="match" and r.matched_invoice_id==gt.expected_invoice_id and r.matched_payment_id==gt.expected_payment_id and r.matched_settlement_id==gt.expected_settlement_id and r.matched_bank_txn_id==gt.expected_bank_txn_id)
        rows.append({"case_id":r.case_id,"scenario":gt.scenario_type if gt else "unknown","terminal_state":r.terminal_state,"correct":correct,
                     "matched_by_ai":r.terminal_state=="matched_by_ai","exception_reason":r.exception_reason})
    scenario=[]
    for scenario_name in SCENARIO_MIX:
        subset=[x for x in rows if x["scenario"]==scenario_name]; resolved=sum(x["terminal_state"] in {"matched_deterministically","matched_by_ai"} for x in subset); correct=sum(x["correct"] for x in subset)
        scenario.append({"scenario":scenario_name,"cases":len(subset),"resolved":resolved,"correct_resolved":correct,"exceptions":len(subset)-resolved,"resolution_rate":resolved/len(subset) if subset else 0,"resolved_accuracy":correct/resolved if resolved else 0})
    return {"run_id":run_id,"ai_mode":next((r.ai_mode for r in results if r.ai_mode),"deterministic-only"),"metrics":asdict(metrics),"false_auto_matches":metrics.false_auto_match_count,
            "ai_contribution_rate":metrics.ai_assisted_resolutions/metrics.total_cases if metrics.total_cases else 0,
            "candidate_quality":_candidate_quality(),
            "scenario_metrics":scenario,
            "exception_breakdown":metrics.exception_breakdown,
            "exceptions":[{"case_id":e.case_id,"reason_code":e.reason_code,"detail":e.detail,"recommended_action":e.recommended_action} for e in exceptions]}


def run_benchmark(seed:int=42, ai_mode:str="local", output_dir:Path|None=None) -> dict[str,Any]:
    init_db(); generate_cases(seed=seed,replace_existing=True)
    baseline=baseline_report()
    run_id,metrics=run_reconciliation(generate_demo=False,ai_mode=ai_mode)
    controller=_controller_report(run_id,metrics)
    report={"seed":seed,"baseline":baseline,"reconpilot":controller,
            "comparison":{"resolution_rate_delta":controller["metrics"]["resolution_rate"]-baseline["resolution_rate"],
                          "resolved_accuracy_delta":controller["metrics"]["resolved_accuracy"]-baseline["resolved_accuracy"],
                          "throughput_delta":controller["metrics"]["throughput_cases_per_second"]-baseline["throughput_cases_per_second"],
                          "false_auto_matches_delta":controller["false_auto_matches"]-baseline["false_auto_matches"]}}
    out=output_dir or BENCHMARK_DIR; out.mkdir(parents=True,exist_ok=True)
    (out/f"run_seed_{seed}.json").write_text(json.dumps(report,indent=2,default=str))
    (out/"metrics.json").write_text(json.dumps({"seed":seed,"baseline":baseline,"reconpilot":controller["metrics"],"comparison":report["comparison"]},indent=2,default=str))
    (out/"scenario_metrics.json").write_text(json.dumps({"baseline":baseline["scenario_metrics"],"reconpilot":controller["scenario_metrics"]},indent=2))
    (out/"exceptions.json").write_text(json.dumps(controller["exceptions"],indent=2))
    return report


def run_multi_seed(seeds:list[int], ai_mode:str="local", output_dir:Path|None=None) -> dict[str,Any]:
    import statistics
    reports=[run_benchmark(seed=x,ai_mode=ai_mode,output_dir=output_dir) for x in seeds]
    ms=[r["reconpilot"]["metrics"] for r in reports]
    aggregate={}
    for key in ("resolution_rate","resolved_accuracy","throughput_cases_per_second","false_auto_match_count"):
        vals=[float(m[key]) for m in ms]
        aggregate[key]={"mean":statistics.mean(vals),"std":statistics.pstdev(vals) if len(vals)>1 else 0.0,"min":min(vals),"max":max(vals)}
    result={"seeds":seeds,"ai_mode":ai_mode,"runs":reports,"aggregate":aggregate}
    out=output_dir or BENCHMARK_DIR; out.mkdir(parents=True,exist_ok=True)
    (out/"multi_seed_summary.json").write_text(json.dumps(result,indent=2,default=str))
    print("\nMulti-seed summary:")
    for k,v in aggregate.items(): print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}, range=[{v['min']:.4f}, {v['max']:.4f}]")
    return result


def print_report(report:dict[str,Any]) -> None:
    b=report["baseline"]; r=report["reconpilot"]["metrics"]
    print("="*68); print("ReconPilot Benchmark"); print("="*68)
    print(f"Seed: {report['seed']}"); print(f"AI mode: {report['reconpilot']['ai_mode']}")
    print(f"{'Metric':30} {'Baseline':>16} {'ReconPilot':>16}")
    print("-"*68)
    print(f"{'Cases':30} {b['total_cases']:>16} {r['total_cases']:>16}")
    print(f"{'Resolved':30} {b['matched']:>16} {r['deterministic_resolutions']+r['ai_assisted_resolutions']:>16}")
    print(f"{'AI resolved':30} {'0':>16} {r['ai_assisted_resolutions']:>16}")
    print(f"{'Suggested':30} {'0':>16} {r['suggested_matches']:>16}")
    print(f"{'Exceptions':30} {b['exceptions']:>16} {r['unresolved_exceptions']:>16}")
    print(f"{'Resolution rate':30} {b['resolution_rate']:>15.2%} {r['resolution_rate']:>15.2%}")
    print(f"{'Resolved accuracy':30} {b['resolved_accuracy']:>15.2%} {r['resolved_accuracy']:>15.2%}")
    print(f"{'False auto-matches':30} {b['false_auto_matches']:>16} {r['false_auto_match_count']:>16}")
    print(f"{'Throughput (cases/sec)':30} {b['throughput_cases_per_second']:>16.2f} {r['throughput_cases_per_second']:>16.2f}")
    print("\nScenario metrics:")
    for x,y in zip(b["scenario_metrics"],report["reconpilot"]["scenario_metrics"]):
        print(f"  {x['scenario']:24} {x['resolution_rate']:6.1%} -> {y['resolution_rate']:6.1%} ({y['correct_resolved']}/{y['cases']} correct)")
    print(f"\nCandidate Recall@1: {report['reconpilot']['candidate_quality']['recall_at_1']:.2%}")
    print(f"Candidate Recall@3: {report['reconpilot']['candidate_quality']['recall_at_3']:.2%}")
    print(f"Reports written to {BENCHMARK_DIR}")
