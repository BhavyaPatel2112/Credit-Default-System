# M7: container for the scoring service. Built in CI on every pull request
# (.github/workflows/gate.yml). Python 3.12: xgboost 3.4 and shap 0.53 need it.
# Build and run locally with:
#   docker build -t cap501-scoring . && docker run -p 8000:8000 cap501-scoring
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api/ api/
COPY src/ src/
COPY models/ models/
COPY runs/ runs/
COPY reports/eda.json reports/eda.json
COPY web/dist/ web/dist/
EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
