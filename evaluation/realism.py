def keyword_hits(text: str, keywords: list[str]) -> int:
    """
    Count how many domain-specific keywords appear in a given text.

    This function performs a simple substring check (case-insensitive)
    for each keyword and counts how many are present at least once.

    It is used as a lightweight realism guardrail to ensure that
    generated reviews reference relevant SaaS task-management concepts
    (e.g. tasks, workflows, integrations).
    """
    text = text.lower()
    return sum(1 for k in keywords if k in text)


def has_drawback(text: str, markers: list[str]) -> bool:
    """
    Check whether a review mentions at least one drawback or limitation.

    The function searches for predefined drawback markers
    (e.g. 'however', 'but', 'although') in a case-insensitive manner.

    This guardrail enforces realism by requiring mild criticism
    in high-rating reviews (4–5 stars), reflecting real user behavior.
    """
    text = text.lower()
    return any(m in text for m in markers)
