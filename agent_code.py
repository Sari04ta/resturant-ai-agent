
import os
from typing import Dict, Any

def build_context(selected_name: str, user_question: str, metrics: Dict[str, Any]) -> str:
    """Build a compact textual context for the LLM from pre-computed metrics."""
    rest = metrics["restaurants"].loc[selected_name]
    sent = metrics["sentiment"].loc[selected_name]
    comp = metrics["competitors"].get(selected_name, None)

    lines = []
    lines.append(f"Restaurant name: {selected_name}")
    lines.append(f"City: {rest['city']}")
    lines.append(f"Cuisine: {rest['cuisine']}")
    lines.append(f"Zone: {rest.get('zone', 'N/A')}")
    lines.append(f"Average rating: {rest['avg_rating']:.2f}")
    lines.append(f"Review count: {rest['review_count']}")
    lines.append(f"Weighted review count: {rest.get('weighted_reviews', rest['review_count']):.1f}")
    lines.append(f"Positive review %: {sent['positive_pct']:.1f}%")
    lines.append(f"Satisfaction score (0-1): {rest['satisfaction_score']:.3f}")
    if 'avg_delivery' in rest:
        lines.append(f"Average delivery time: {rest['avg_delivery']:.1f} minutes")
    if 'price_range' in rest:
        lines.append(f"Typical price range: {rest['price_range']}")
    if 'avg_popularity' in rest:
        lines.append(f"Average menu popularity score (0-100): {rest['avg_popularity']:.1f}")

    if comp is not None and not comp.empty:
        lines.append("\nTop competitors in same city & cuisine (by satisfaction score):")
        top_comp = comp.head(5)
        for _, row in top_comp.iterrows():
            lines.append(
                f"- {row.name} — rating {row['avg_rating']:.2f}, "
                f"satisfaction {row['satisfaction_score']:.3f}, "
                f"review_count {row['review_count']}, "
                f"avg_delivery {row.get('avg_delivery', float('nan')):.1f} min"
            )

    lines.append("\nUser question:")
    lines.append(user_question.strip())

    return "\n".join(lines)


def _call_openai(prompt: str, temperature: float = 0.3) -> str:
    """Call OpenAI Chat Completions API. Requires OPENAI_API_KEY env var."""
    try:
        from openai import OpenAI
    except ImportError:
        return (
            "OpenAI Python package not installed. "
            "Install `openai` and set OPENAI_API_KEY to use this provider."
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "OPENAI_API_KEY environment variable not set. "
            "Please configure it on the server."
        )

    client = OpenAI(api_key=api_key)
    system = (
        "You are an expert restaurant business analyst. "
        "You receive structured summary data about a single restaurant and its competitors. "
        "Provide clear, actionable insights and recommendations. "
        "Do not fabricate numbers; use qualitative language when exact values are not provided."
    )
    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return res.choices[0].message.content


def _call_huggingface(prompt: str, temperature: float = 0.3) -> str:
    """Call a HuggingFace text generation model. Requires HF_API_TOKEN env var."""
    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        return (
            "huggingface_hub package not installed. "
            "Install it and set HF_API_TOKEN to use this provider."
        )

    token = os.getenv("HF_API_TOKEN")
    if not token:
        return (
            "HF_API_TOKEN environment variable not set. "
            "Please configure it on the server."
        )

    client = InferenceClient(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        token=token,
    )
    response = client.text_generation(
        prompt,
        max_new_tokens=512,
        temperature=float(temperature),
    )
    return response


def run_agent(
    user_question: str,
    context: str,
    provider: str = "openai",
    temperature: float = 0.3,
) -> str:
    """Dispatch the question + context to the chosen LLM provider."""
    prompt = (
        "You are given analytics about a restaurant and its competitors.\n"
        "Use only the information in the context.\n\n"
        f"CONTEXT:\n{context}\n\n"
        "TASK: Answer the user's question with clear, structured, practical recommendations. "
        "Highlight strengths, weaknesses, and 3–5 concrete action items."
    )
    if provider == "huggingface":
        return _call_huggingface(prompt, temperature=temperature)
    return _call_openai(prompt, temperature=temperature)
