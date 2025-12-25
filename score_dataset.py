import json
import yaml
from tqdm import tqdm

from evaluation.sentiment import rating_sentiment_ok
from evaluation.diversity import vocab_overlap, too_similar_embedding
from evaluation.realism import keyword_hits, has_drawback
from models.openai_model import generate_embedding

cfg = yaml.safe_load(open("config.yaml"))

dataset_path = cfg["outputs"]["dataset_path"]
output_path = "outputs/synthetic_reviews_scored.jsonl"

reviews = []
with open(dataset_path, "r") as f:
    for line in f:
        reviews.append(json.loads(line))

embeddings = []

scored_reviews = []

for r in tqdm(reviews):
    checks_passed = 0
    total_checks = 5

    review_text = r["review"]
    rating = r["rating"]

    if rating_sentiment_ok(
        review_text,
        rating,
        cfg["guardrails"]["sentiment"]["low_rating_positive_cutoff"],
        cfg["guardrails"]["sentiment"]["high_rating_negative_cutoff"],
    ):
        checks_passed += 1

    if keyword_hits(review_text, cfg["domain"]["keywords"]) >= cfg["guardrails"]["realism"]["min_keyword_hits"]:
        checks_passed += 1

    drawback_ok = True
    if rating >= 4 and cfg["guardrails"]["realism"]["require_drawback_for_high_ratings"]:
        drawback_ok = has_drawback(review_text, cfg["guardrails"]["realism"]["drawback_markers"])

    if drawback_ok:
        checks_passed += 1

    emb = generate_embedding(review_text, cfg["embeddings"]["model"])
    if not too_similar_embedding(
        emb,
        embeddings,
        cfg["guardrails"]["semantic_similarity"]["threshold"],
    ):
        checks_passed += 1

    embeddings.append(emb)

    vocab_ok = True
    for prev in scored_reviews:
        if vocab_overlap(review_text, prev["review"]) > cfg["guardrails"]["vocabulary_overlap"]["threshold"]:
            vocab_ok = False
            break

    if vocab_ok:
        checks_passed += 1

    quality_score = round(checks_passed / total_checks, 2)

    r["quality_score"] = quality_score
    scored_reviews.append(r)

with open(output_path, "w") as f:
    for r in scored_reviews:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("Scoring complete.")

