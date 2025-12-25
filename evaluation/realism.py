def keyword_hits(text: str, keywords: list[str]) -> int:
    text = text.lower()
    return sum(1 for k in keywords if k in text)


def has_drawback(text: str, markers: list[str]) -> bool:
    text = text.lower()
    return any(m in text for m in markers)
