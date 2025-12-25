from sklearn.metrics.pairwise import cosine_similarity

def vocab_overlap(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def too_similar_embedding(new_vec, existing_vecs, threshold: float) -> bool:
    for vec in existing_vecs:
        sim = cosine_similarity([new_vec], [vec])[0][0]
        if sim >= threshold:
            return True
    return False
