from __future__ import annotations
import json
from decimal import Decimal, InvalidOperation
from typing import Any

REQUIRED_RECORDS = ("invoice", "payment", "settlement", "bank_txn")

def validate_payload(payload: dict[str, Any]) -> list[dict[str,str]]:
    errors=[]
    for key in REQUIRED_RECORDS:
        if key not in payload or payload[key] is None:
            errors.append({"code":"MISSING_RECORD","field":key,"message":f"Missing {key} record"})
    for section,field in (("invoice","amount"),("payment","amount"),("settlement","net_amount"),("bank_txn","amount")):
        obj=payload.get(section)
        if isinstance(obj,dict) and obj.get(field) is not None:
            try: Decimal(str(obj[field]))
            except (InvalidOperation,ValueError): errors.append({"code":"INVALID_RECORD","field":f"{section}.{field}","message":"Amount is not numeric"})
    return errors

def validate_json_file(path:str) -> list[dict[str,str]]:
    with open(path,encoding="utf-8") as f: payload=json.load(f)
    return validate_payload(payload)
