import hashlib
import json
from typing import List, Dict, Optional
from .schema import CanonicalRecord, Message


def _make_id(messages: list) -> str:
    raw = json.dumps(messages, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _from_messages(record: dict) -> CanonicalRecord:
    msgs = [Message(role=m["role"], content=m["content"]) for m in record["messages"]]
    return CanonicalRecord(id=_make_id(record["messages"]), messages=msgs)


def _from_sharegpt(record: dict) -> CanonicalRecord:
    role_map = {"human": "user", "gpt": "assistant", "system": "system"}
    msgs = []
    for turn in record.get("conversations", []):
        role = role_map.get(turn.get("from", ""), turn.get("from", "user"))
        msgs.append(Message(role=role, content=turn.get("value", "")))
    return CanonicalRecord(id=_make_id(record.get("conversations", [])), messages=msgs)


def _from_alpaca(record: dict) -> CanonicalRecord:
    user_text = record.get("instruction", "")
    if record.get("input", "").strip():
        user_text += "\n\n" + record["input"]
    msgs = [
        Message(role="user", content=user_text),
        Message(role="assistant", content=record.get("output", "")),
    ]
    raw = [{"role": m.role, "content": m.content} for m in msgs]
    return CanonicalRecord(id=_make_id(raw), messages=msgs)


def _from_csv_custom(record: dict, column_map: dict) -> CanonicalRecord:
    msgs = []
    if column_map.get("system") and record.get(column_map["system"]):
        msgs.append(Message(role="system", content=record[column_map["system"]]))
    msgs.append(Message(role="user", content=record.get(column_map["user"], "")))
    msgs.append(
        Message(role="assistant", content=record.get(column_map["assistant"], ""))
    )
    raw = [{"role": m.role, "content": m.content} for m in msgs]
    return CanonicalRecord(id=_make_id(raw), messages=msgs)


def _from_completion(record: dict) -> CanonicalRecord:
    text = record.get("text", "")
    return CanonicalRecord(id=_make_id([text]), messages=[], meta={"text": text})


def _detect_format(record: dict) -> str:
    if "messages" in record:
        return "messages"
    if "conversations" in record:
        return "sharegpt"
    if "instruction" in record and "output" in record:
        return "alpaca"
    if "text" in record:
        return "completion"
    return "csv_custom"


def normalize(
    records: List[Dict], fmt: str = "auto", column_map: Optional[Dict] = None
) -> List[CanonicalRecord]:
    out = []
    for rec in records:
        effective_fmt = _detect_format(rec) if fmt == "auto" else fmt
        try:
            if effective_fmt == "messages":
                out.append(_from_messages(rec))
            elif effective_fmt == "sharegpt":
                out.append(_from_sharegpt(rec))
            elif effective_fmt == "alpaca":
                out.append(_from_alpaca(rec))
            elif effective_fmt == "csv_custom":
                out.append(_from_csv_custom(rec, column_map or {}))
            elif effective_fmt == "completion":
                out.append(_from_completion(rec))
        except Exception:
            pass
    return out
