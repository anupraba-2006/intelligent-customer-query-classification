
HIGH_PRIORITY_KEYWORDS = [
    "urgent", "asap", "immediately", "escalate", "escalating", "manager",
    "cancelling", "cancel my order", "refund", "not working", "stopped working",
    "damaged", "defective", "rusted", "broken", "tender", "stalled",
]

MEDIUM_PRIORITY_KEYWORDS = [
    "delayed", "still waiting", "any update", "follow up", "following up",
]

PRIORITY_RANK = {"Low": 0, "Medium": 1, "High": 2}


def rule_override(text: str, model_priority: str) -> tuple[str, bool]:
    """
    Returns (final_priority, was_overridden).
    Only escalates priority upward based on keyword matches; never downgrades
    a model prediction.
    """
    text_lower = (text or "").lower()
    suggested = model_priority

    if any(kw in text_lower for kw in HIGH_PRIORITY_KEYWORDS):
        suggested = "High"
    elif any(kw in text_lower for kw in MEDIUM_PRIORITY_KEYWORDS):
        if PRIORITY_RANK.get(model_priority, 0) < PRIORITY_RANK["Medium"]:
            suggested = "Medium"

    was_overridden = suggested != model_priority
    return suggested, was_overridden
