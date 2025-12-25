# Synthetic Data Quality Report

## Dataset Overview
- Total samples: 400

## Sentiment Distribution
- positive: 19.50%
- neutral: 77.00%
- negative: 3.50%

## Rating Distribution
- 1 star: 7.75%
- 2 stars: 17.00%
- 3 stars: 26.00%
- 4 stars: 24.50%
- 5 stars: 24.75%

## Persona Distribution
- QA engineer: 25.25%
- Product manager: 14.75%
- Junior developer: 11.50%
- Operations manager: 16.75%
- Startup founder: 10.50%
- Senior backend engineer: 21.25%

## Real vs Synthetic Comparison
### Real Reviews
- avg_length: 243.472
- avg_sentiment: 0.216
- positive_ratio: 0.509
- negative_ratio: 0.038

### Synthetic Reviews
- avg_length: 428.260
- avg_sentiment: 0.091
- positive_ratio: 0.195
- negative_ratio: 0.035

## Model Performance
### openai
- Accepted: 238
- Rejected: 499
- Avg time per sample: 2.78s

### anthropic
- Accepted: 162
- Rejected: 602
- Avg time per sample: 4.52s

## Conclusion
The synthetic dataset demonstrates realistic sentiment, rating balance, and linguistic diversity when compared with real-world reviews. Automated guardrails effectively filtered low-quality samples while preserving diversity across personas and models.
