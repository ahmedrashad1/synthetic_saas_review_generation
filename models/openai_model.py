import os
import json
from openai import OpenAI
from typing import Dict, Optional

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_review(prompt: str, model: str, temperature: float, max_tokens: int) -> Optional[Dict]:
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
   
    emb = client.embeddings.create(
        model=model,
        input=text
    )
    return emb.data[0].embedding


def _parse_json(text: str) -> Optional[Dict]:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(text[start:end + 1])
    except Exception:
        return None
