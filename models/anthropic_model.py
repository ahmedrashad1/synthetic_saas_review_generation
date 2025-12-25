import os
import json
from anthropic import Anthropic
from typing import Dict, Optional

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def generate_review(prompt: str, model: str, temperature: float, max_tokens: int) -> Optional[Dict]:
    """
    Generate a single synthetic SaaS review using an Anthropic Claude model.

    The function sends a user-only prompt to the Anthropic Messages API
    instructing the model to produce a realistic product review in
    STRICT JSON format.

    The returned JSON is expected to include fields such as:
    - persona
    - rating
    - review
    """
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
