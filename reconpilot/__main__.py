from __future__ import annotations
import argparse
from reconpilot.benchmark import run_benchmark, print_report, run_multi_seed
from reconpilot.generate import generate_cases, get_scenario_counts
from reconpilot.pipeline import run_reconciliation

def main():
    parser=argparse.ArgumentParser(description="ReconPilot finance reconciliation controller")
    sub=parser.add_subparsers(dest="command")
    runp=sub.add_parser("run",help="run reconciliation")
    runp.add_argument("--seed",type=int,default=42); runp.add_argument("--mode",choices=["local","live","off"],default="local")
    bp=sub.add_parser("benchmark",help="generate and evaluate baseline vs ReconPilot")
    bp.add_argument("--seed",type=int,default=42); bp.add_argument("--seeds",type=int,nargs="+",default=None); bp.add_argument("--mode",choices=["local","live","off"],default="local")
    gp=sub.add_parser("generate",help="generate synthetic benchmark data")
    gp.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    command=args.command or "run"
    if command=="generate":
        ids=generate_cases(seed=args.seed,replace_existing=True); print(f"Generated {len(ids)} cases"); print(get_scenario_counts()); return
    if command=="benchmark":
        if args.seeds:
            run_multi_seed(args.seeds,ai_mode=args.mode)
        else:
            report=run_benchmark(seed=args.seed,ai_mode=args.mode); print_report(report)
        return
    run_id,metrics=run_reconciliation(generate_demo=True,seed=args.seed,ai_mode=args.mode)
    print("="*60); print("ReconPilot - Reconciliation Run"); print("="*60)
    print(f"Run ID: {run_id}"); print(f"AI mode: {args.mode}"); print(f"Total cases: {metrics.total_cases}")
    print(f"Deterministic resolutions: {metrics.deterministic_resolutions}"); print(f"AI-assisted resolutions: {metrics.ai_assisted_resolutions}")
    print(f"Suggested matches: {metrics.suggested_matches}"); print(f"Unresolved exceptions: {metrics.unresolved_exceptions}")
    print(f"Resolution rate: {metrics.resolution_rate:.2%}"); print(f"Resolved accuracy: {metrics.resolved_accuracy:.2%}")
    print(f"Overall accuracy: {metrics.overall_accuracy:.2%}"); print(f"False auto-matches: {metrics.false_auto_match_count}")
    print(f"Throughput: {metrics.throughput_cases_per_second:.2f} cases/sec")
    if metrics.exception_breakdown:
        print("Exception breakdown:"); [print(f"  - {k}: {v}") for k,v in metrics.exception_breakdown.items()]

if __name__=="__main__": main()
