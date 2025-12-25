from sklearn.metrics.pairwise import cosine_similarity

def vocab_overlap(a: str, b: str) -> float:
    """
    Compute the lexical overlap ratio between two text strings.

    The texts are tokenized into lowercase word sets.
    The overlap is measured using Jaccard similarity:

        overlap = |A ∩ B| / |A ∪ B|

    This metric is used as a lightweight guardrail to detect
    excessive word reuse between reviews and encourage
    surface-level linguistic diversity.
    """
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def too_similar_embedding(new_vec, existing_vecs, threshold: float) -> bool:
    """
    Determine whether a new embedding vector is too similar
    to any previously accepted embedding.

    Cosine similarity is computed between the new vector
    and each existing vector. If any similarity exceeds
    the given threshold, the review is considered a near-duplicate
    and should be rejected.

    This guardrail prevents semantic duplication even when
    surface wording differs.
    """
    for vec in existing_vecs:
        sim = cosine_similarity([new_vec], [vec])[0][0]
        if sim >= threshold:
            return True
    return False
