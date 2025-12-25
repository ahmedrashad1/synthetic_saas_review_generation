import os
import json
from openai import OpenAI
from typing import Dict, Optional

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_review(prompt: str, model: str, temperature: float, max_tokens: int) -> Optional[Dict]:
    """
    Generate a single synthetic SaaS review using openai chat model.

    The function sends a user-only prompt to the openai Messages API
    instructing the model to produce a realistic product review in
    STRICT JSON format.

    The returned JSON is expected to include fields such as:
    - persona
    - rating
    - review
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You generate realistic SaaS product reviews."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    text = response.choices[0].message.content.strip()

    return _parse_json(text)

def generate_embedding(text: str, model: str) -> list[float]:
    """
    Generate a semantic embedding vector for a given text using an OpenAI
    embedding model.

    The resulting vector represents the semantic meaning of the input text
    and is used to measure similarity between reviews via cosine similarity.
    This enables detection of near-duplicate or highly similar synthetic
    samples during both generation and post-generation scoring.
    """
    emb = client.embeddings.create(
        model=model,
        input=text
    )
    return emb.data[0].embedding


def _parse_json(text: str) -> Optional[Dict]:
    """
    Safely extract and parse a JSON object from raw model output.

    This helper function searches for the first opening brace '{'
    and the last closing brace '}' to isolate a JSON object,
    then attempts to parse it.

    The function is intentionally defensive and will return None
    rather than raising an exception if parsing fails.
    """
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(text[start:end + 1])
    except Exception:
        return None
