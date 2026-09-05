# ReconPilot Submission Validation Report

## Build status

- Automated tests: **73 passed**
- Synthetic benchmark size: **104 cases**
- Scenario coverage: **10 controlled scenarios**
- AI modes: `local`, `live`, `off`
- API: FastAPI endpoints implemented
- UI: Streamlit dashboard with KPI, scenario, exception, investigation and audit views
- Reproducible benchmark artifacts: included under `benchmark/`
- Secrets/database/virtual environments: excluded from submission archive

## Seed 42 benchmark

| Metric | Deterministic baseline | ReconPilot local evaluator |
|---|---:|---:|
| Cases | 104 | 104 |
| Resolved | 11 | 75 |
| AI-assisted resolved | 0 | 19 |
| Suggested | 0 | 0 |
| Exceptions | 93 | 29 |
| Resolution rate | 10.58% | 72.12% |
| Resolved accuracy | 100.00% | 100.00% |
| False auto-matches | 0 | 0 |
| Throughput | 19,848.83 cases/sec | 29.54 cases/sec |
| Candidate Recall@1 | — | 100.00% |
| Candidate Recall@3 | — | 100.00% |

The baseline is deliberately conservative and performs deterministic reconciliation without candidate ranking or AI. The controller uses the same generated dataset and adds candidate generation plus the explicit local evaluation mode for offline reproducibility.

## Scenario results — seed 42

| Scenario | Cases | ReconPilot resolved | Correct resolved |
|---|---:|---:|---:|
| Exact match | 20 | 20 | 20 |
| Fee-adjusted match | 15 | 15 | 15 |
| Delayed settlement | 10 | 10 | 10 |
| Fuzzy reference | 10 | 10 | 10 |
| Duplicate candidate | 10 | 10 | 10 |
| Partial payment | 8 | 0 | 0 |
| Missing counterparty | 8 | 0 | 0 |
| Corrupt record | 5 | 0 | 0 |
| No valid match | 8 | 0 | 0 |
| Ambiguous match | 10 | 10 | 10 |

## Safety validation

The benchmark records **0 false automatic matches**. AI-selected candidates must come from deterministic candidate generation and must independently pass settlement-to-bank deterministic validation. Ambiguous/uncertain AI classifications are abstained to a review exception. Invalid AI candidate IDs remain a safety block.

## Failure handling

Malformed numeric data is classified as `INVALID_RECORD`. Missing source data is classified as `MISSING_RECORD`. Provider outages are classified as `AI_UNAVAILABLE`/provider-specific failure and terminate safely. AI-disabled mode uses `AI_DISABLED` and does not pretend that a local result came from an LLM.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
python -m reconpilot benchmark --seed 42 --mode local
python -m reconpilot benchmark --seeds 42 43 44 45 46 --mode local
streamlit run app.py
```

For a real provider-backed run, set `RECONPILOT_AI_MODE=live`, configure `LITELLM_MODEL`, and provide the provider credential through the deployment environment. Never commit `.env`.

## Multi-seed validation

Seeds 42–46: resolution rate mean **72.12%** (std 0.00%), resolved accuracy mean **100%**, false auto-match count mean **0**, and controller throughput mean **28.52 cases/sec** (std 0.76). These measurements are from the reproducible local evaluator and should be described as synthetic/offline benchmark results.
