import os
import json
from anthropic import Anthropic
from typing import Dict, Optional

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def generate_review(prompt: str, model: str, temperature: float, max_tokens: int) -> Optional[Dict]:

    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": (
                    "You generate realistic SaaS product reviews.\n\n"
                    + prompt
                )
            }
        ]
    )

    text = message.content[0].text.strip()
    return _safe_parse_json(text)


def _safe_parse_json(text: str) -> Optional[Dict]:
    
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        return json.loads(text[start:end + 1])
    except Exception:
        return None
