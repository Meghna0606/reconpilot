PROJECT: ReconPilot

GOAL:
Build an AI Finance Controller that reconciles invoices,
payments, settlements and bank transactions.

CORE PRINCIPLE:
Use deterministic logic for obvious matches.
Use AI only for ambiguous cases.
Never allow AI to silently force a match.

REQUIRED OUTPUT:
- 50+ record batch
- Throughput metric
- Match rate
- Measured accuracy against ground truth
- Honest exception list
- Audit trail
- Graceful failure recovery

TECH:
Python
FastAPI
Streamlit
SQLite
Pandas
RapidFuzz
Pydantic

RULES:
1. Every decision must be auditable.
2. AI output must be structured.
3. Invalid AI output must not crash the batch.
4. Low confidence must become an exception.
5. No automatic match if conflicting evidence exists.
