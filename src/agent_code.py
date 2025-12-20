import os
import streamlit as st

# -------- Build Context (unchanged) -------- #

def build_context(selected_name: str, user_question: str, metrics):
    rest = metrics["restaurants"].loc[selected_name]
    sent = metrics["sentiment"].loc[selected_name]
    comp = metrics["competitors"].get(selected_name, None)

    lines = []
    lines.append(f"Restaurant: {selected_name}")
    lines.append(f"City: {rest['city']}")
    lines.append(f"Cuisine: {rest['cuisine']}")
    lines.append(f"Zone: {rest.get('zone', 'N/A')}")
    lines.append(f"Average rating: {rest['avg_rating']:.2f}")
    lines.append(f"Review count: {rest['review_count']}")
    lines.append(f"Weighted review count: {rest.get('weighted_reviews', rest['review_count']):.1f}")
    lines.append(f"Positive review %: {sent['positive_pct']:.1f}%")
    lines.append(f"Satisfaction score (0–1): {rest['satisfaction_score']:.3f}")

    if "avg_delivery" in rest:
        lines.append(f"Average delivery time: {rest['avg_delivery']:.1f} minutes")

    if "price_range" in rest:
        lines.append(f"Price range: {rest['price_range']}")

    if "avg_popularity" in rest:
        lines.append(f"Menu popularity score: {rest['avg_popularity']:.1f}")

    if comp is not None and not comp.empty:
        lines.append("\nTop competitors:")
        for _, row in comp.head(5).iterrows():
            lines.append(
                f"- {row.name} — rating {row['avg_rating']:.2f}, "
                f"satisfaction {row['satisfaction_score']:.3f}, "
                f"delivery {row.get('avg_delivery', float('nan')):.1f} min"
            )

    lines.append("\nUser question:")
    lines.append(user_question)

    return "\n".join(lines)


# -------- OpenAI Provider (Now uses Streamlit Secrets) -------- #

def _call_openai(prompt: str, temperature: float = 0.3) -> str:
    try:
        from openai import OpenAI
    except:
        return "OpenAI package not installed."

    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return "❌ OPENAI_API_KEY missing in .streamlit/secrets.toml"

    client = OpenAI(api_key=api_key)

    system_prompt = (
        "You are an expert restaurant business analyst. "
        "Provide insights based only on the given context. "
        "Give clear recommendations and avoid guessing values."
    )

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",

            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"❌ OpenAI API Error: {e}"


# -------- HuggingFace Provider (Now uses Streamlit Secrets) -------- #

def _call_hf(prompt: str, temperature: float = 0.3) -> str:
    try:
        from huggingface_hub import InferenceClient
    except:
        return "huggingface_hub package not installed."

    token = st.secrets.get("HF_API_TOKEN")
    if not token:
        return "❌ HF_API_TOKEN missing in .streamlit/secrets.toml"

    try:
        client = InferenceClient(
            model="mistralai/Mixtral-8x7B-Instruct-v0.1",
            token=token
        )
        response = client.text_generation(
            prompt,
            max_new_tokens=400,
            temperature=float(temperature)
        )
        return response
    except Exception as e:
        return f"❌ HuggingFace API Error: {e}"


# -------- Main Agent Caller -------- #

def run_agent(user_question: str, context: str, provider="openai", temperature=0.3):
    prompt = (
        "CONTEXT:\n"
        + context
        + "\n\nTASK: Based on the context above, answer the user's question with clear analysis, "
          "strengths, weaknesses, and 3–5 specific business recommendations."
    )

    if provider.lower() == "huggingface":
        return _call_hf(prompt, temperature)

    return _call_openai(prompt, temperature)
