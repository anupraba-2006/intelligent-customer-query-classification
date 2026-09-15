
import re

_WHITESPACE_RE = re.compile(r"\s+")
_ORDER_ID_RE = re.compile(r"\bORD-\d{3,6}\b", flags=re.IGNORECASE)
_QUANTITY_RE = re.compile(r"\b(\d{1,6})\s*(units?|pcs?|pieces?|nos\.?)?\b", flags=re.IGNORECASE)


def clean_text(text: str) -> str:
    """Lowercase, trim, and normalize whitespace. Keeps punctuation/numbers
    since order IDs and quantities are meaningful signals for this domain."""
    if text is None:
        return ""
    text = str(text).strip()
    text = _WHITESPACE_RE.sub(" ", text)
    return text.lower()


def extract_order_id(text: str) -> str | None:
    match = _ORDER_ID_RE.search(text or "")
    return match.group(0).upper() if match else None


def extract_quantity(text: str) -> int | None:
    """Best-effort extraction of a quantity mentioned in the query.
    Not exhaustive by design -- used for response templating, not as a
    hard requirement. Returns the first plausible integer found."""
    for match in _QUANTITY_RE.finditer(text or ""):
        num = match.group(1)
        if num and num.isdigit():
            value = int(num)
            # heuristic: ignore obvious order-id fragments / years
            if 0 < value < 1_000_000:
                return value
    return None
