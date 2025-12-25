import time
import json
import random
import yaml
import os
from tqdm import tqdm

from models.openai_model import generate_review as openai_generate
from models.openai_model import generate_embedding
from models.anthropic_model import generate_review as anthropic_generate

from evaluation.sentiment import rating_sentiment_ok
from evaluation.diversity import vocab_overlap, too_similar_embedding
from evaluation.realism import keyword_hits, has_drawback

from analysis.bias_analysis import (
    analyze_sentiment,
    analyze_ratings,
    analyze_personas,
)

from analysis.real_comparison import (
    load_real_reviews,
    compare_real_vs_synthetic,
)

from analysis.report import generate_report


def weighted_choice(distribution: dict) -> int:
    r = random.random()
    cumulative = 0.0
    for rating, prob in distribution.items():
        cumulative += prob
        if r <= cumulative:
            return int(rating)
    return int(list(distribution.keys())[-1])


def build_prompt(domain, persona, style_notes, rating, min_chars, max_chars):
    return f"""
Generate ONE realistic review for a {domain}.

Persona: {persona}
Persona style: {style_notes}
Rating: {rating} out of 5

Rules:
- Write like a real human user, not marketing.
- Mention at least two concrete product features.
- If rating is 4 or 5, include at least one drawback.
- Keep length between {min_chars} and {max_chars} characters.
- Output STRICT JSON with keys: persona, rating, review.
- No text outside JSON.
""".strip()


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    os.makedirs(os.path.dirname(cfg["outputs"]["dataset_path"]), exist_ok=True)
    target = cfg["generation"]["target_accepted"]
    max_attempts = cfg["generation"]["max_attempts"]

    accepted = []
    embeddings = []

    model_stats = {
        "openai": {"accepted": 0, "rejected": 0, "time": 0.0},
        "anthropic": {"accepted": 0, "rejected": 0, "time": 0.0},
    }

    attempts = 0
    pbar = tqdm(total=target)

    while len(accepted) < target and attempts < max_attempts:
        attempts += 1

        model_cfg = random.choice(cfg["models"])
        model_provider = model_cfg["provider"]

        persona_cfg = random.choice(cfg["generation"]["personas"])
        rating = weighted_choice(cfg["generation"]["rating_distribution"])

        prompt = build_prompt(
            cfg["domain"]["name"],
            persona_cfg["name"],
            persona_cfg["style_notes"],
            rating,
            cfg["generation"]["min_chars"],
            cfg["generation"]["max_chars"],
        )

        start = time.time()

        if model_provider == "openai":
            result = openai_generate(
                prompt,
                model_cfg["model"],
                model_cfg["temperature"],
                model_cfg["max_tokens"],
            )
        else:
            result = anthropic_generate(
                prompt,
                model_cfg["model"],
                model_cfg["temperature"],
                model_cfg["max_tokens"],
            )

        elapsed = time.time() - start
        model_stats[model_provider]["time"] += elapsed

        if not result or "review" not in result:
            model_stats[model_provider]["rejected"] += 1
            continue

        review_text = result["review"]

        # ---- Guardrails ----

        # Sentiment vs rating
        if not rating_sentiment_ok(
            review_text,
            rating,
            cfg["guardrails"]["sentiment"]["low_rating_positive_cutoff"],
            cfg["guardrails"]["sentiment"]["high_rating_negative_cutoff"],
        ):
            model_stats[model_provider]["rejected"] += 1
            continue

        # Domain realism
        if keyword_hits(review_text, cfg["domain"]["keywords"]) < cfg["guardrails"]["realism"]["min_keyword_hits"]:
            model_stats[model_provider]["rejected"] += 1
            continue

        if rating >= 4 and cfg["guardrails"]["realism"]["require_drawback_for_high_ratings"]:
            if not has_drawback(review_text, cfg["guardrails"]["realism"]["drawback_markers"]):
                model_stats[model_provider]["rejected"] += 1
                continue

        # Embedding similarity
        new_embedding = generate_embedding(
            review_text,
            cfg["embeddings"]["model"]
        )

        if too_similar_embedding(
            new_embedding,
            embeddings,
            cfg["guardrails"]["semantic_similarity"]["threshold"],
        ):
            model_stats[model_provider]["rejected"] += 1
            continue

        # Vocabulary overlap
        if any(
            vocab_overlap(review_text, r["review"]) > cfg["guardrails"]["vocabulary_overlap"]["threshold"]
            for r in accepted
        ):
            model_stats[model_provider]["rejected"] += 1
            continue

        # ---- ACCEPT ----
        record = {
            "model": model_provider,
            "persona": persona_cfg["name"],
            "rating": rating,
            "review": review_text,
        }

        accepted.append(record)
        embeddings.append(new_embedding)
        model_stats[model_provider]["accepted"] += 1
        pbar.update(1)

    pbar.close()

    
    with open(cfg["outputs"]["dataset_path"], "w") as f:
        for r in accepted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    
    with open(cfg["outputs"]["run_log_path"], "w") as f:
        json.dump(model_stats, f, indent=2)

    print("Generation complete.")
    print(model_stats)

    
    sentiment_stats = analyze_sentiment(accepted)
    rating_stats = analyze_ratings(accepted)
    persona_stats = analyze_personas(accepted)

    
    real_reviews = load_real_reviews(cfg["outputs"]["real_reviews_path"])

    
    comparison = compare_real_vs_synthetic(real_reviews, accepted)

    
    generate_report(
        sentiment_stats=sentiment_stats,
        rating_stats=rating_stats,
        persona_stats=persona_stats,
        comparison=comparison,
        model_stats=model_stats,
        output_path=cfg["outputs"]["report_path"],
    )


if __name__ == "__main__":
    main()
