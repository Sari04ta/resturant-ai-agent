
# AI Agent — Restaurant Market Analysis

## 1. Project structure

- `app.py` — Streamlit UI (dashboards + AI agent)
- `agent_code.py` — LLM context builder + provider calls (OpenAI / HuggingFace)
- `utils/data_utils.py` — CSV loading & preprocessing
- `utils/analysis_utils.py` — metrics, charts, competitor logic
- `requirements.txt` — Python dependencies
- `Dockerfile` — containerised deployment

## 2. Expected CSV schema

Required columns (case-insensitive):

- `name`
- `city`
- `cuisine`
- `rating`
- `review_text`
- `review_date`
- `lat`
- `lon`

Optional (used if present):

- `zone`
- `num_reviews`
- `price_range`
- `delivery_time`
- `menu_item_popularity`

## 3. Local run

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="your-openai-key"      # optional
export HF_API_TOKEN="your-hf-token"          # optional

streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```



## 4. Docker

```bash
docker build -t restaurant-agent .
docker run -p 8501:8501 \
  -e OPENAI_API_KEY="your-openai-key" \
  -e HF_API_TOKEN="your-hf-token" \
  restaurant-agent
```

