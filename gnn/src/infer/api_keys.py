from __future__ import annotations


def api_key_from_hf(api: dict | list) -> str:
    if isinstance(api, (list, tuple)):
        if len(api) >= 3:
            return f"{api[0]}::{api[1]}::{api[2]}"
        if len(api) >= 2:
            return f"{api[0]}::{api[0]}::{api[1]}"
        return str(api[0]) if api else ""
    cat = api.get("category_name") or api.get("category") or ""
    tool = api.get("tool_name") or api.get("tool") or ""
    api_name = api.get("api_name") or api.get("api") or ""
    return f"{cat}::{tool}::{api_name}"


def api_match_key(key: str, other: str) -> bool:
    if key == other:
        return True
    a = key.split("::")
    b = other.split("::")
    if len(a) >= 3 and len(b) >= 2:
        return a[1] == b[0] or a[1] == b[1] or a[2] == b[-1]
    return key in other or other in key
