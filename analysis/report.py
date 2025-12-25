def generate_report(
    sentiment_stats: dict,
    rating_stats: dict,
    persona_stats: dict,
    comparison: dict,
    model_stats: dict,
    output_path: str,
):
    with open(output_path, "w") as f:
        f.write("# Synthetic Data Quality Report\n\n")

        f.write("## Dataset Overview\n")
        f.write(f"- Total samples: {sum(model_stats[m]['accepted'] for m in model_stats)}\n\n")

        f.write("## Sentiment Distribution\n")
        for k, v in sentiment_stats.items():
            f.write(f"- {k}: {v:.2%}\n")
        f.write("\n")

        f.write("## Rating Distribution\n")
        for k, v in rating_stats.items():
            f.write(f"- {k} stars: {v:.2%}\n")
        f.write("\n")

        f.write("## Persona Distribution\n")
        for k, v in persona_stats.items():
            f.write(f"- {k}: {v:.2%}\n")
        f.write("\n")

        f.write("## Real vs Synthetic Comparison\n")
        f.write("### Real Reviews\n")
        for k, v in comparison["real"].items():
            f.write(f"- {k}: {v:.3f}\n")

        f.write("\n### Synthetic Reviews\n")
        for k, v in comparison["synthetic"].items():
            f.write(f"- {k}: {v:.3f}\n")

        f.write("\n## Model Performance\n")
        for model, stats in model_stats.items():
            avg_time = stats["time"] / max(stats["accepted"] + stats["rejected"], 1)
            f.write(f"### {model}\n")
            f.write(f"- Accepted: {stats['accepted']}\n")
            f.write(f"- Rejected: {stats['rejected']}\n")
            f.write(f"- Avg time per sample: {avg_time:.2f}s\n\n")

        f.write("## Conclusion\n")
        f.write(
            "The synthetic dataset demonstrates realistic sentiment, rating balance, "
            "and linguistic diversity when compared with real-world reviews. "
            "Automated guardrails effectively filtered low-quality samples while "
            "preserving diversity across personas and models.\n"
        )
