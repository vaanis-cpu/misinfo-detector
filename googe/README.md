# Misinformation Graph Detector

Real-time misinformation detection system using propagation graphs, GNN, and temporal prediction.

## Architecture

- **Ingestion Layer**: Stream posts from social APIs (Twitter, Reddit, Facebook)
- **Streaming Layer**: Kafka + Flink for event processing
- **NLP Encoding**: DeBERTa/RoBERTa for claim embeddings
- **Propagation GNN**: GraphSAGE/GAT for learning viral patterns
- **Temporal Predictor**: LSTM on graph snapshots for virality prediction
- **Output**: Risk scores, API, SHAP explainability, dashboard

## Quick Start

### Using Docker (Recommended)

```bash
# Development stack (lightweight)
docker-compose -f docker-compose.dev.yml up

# Full production stack
docker-compose up
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run MVP
./scripts/run_mvp.sh

# Or manually:
python scripts/seed_data.py
uvicorn src.api.main:app --reload
streamlit run src/dashboard/app.py
```

## API Endpoints

- `POST /claims` - Submit a claim for assessment
- `GET /claims/{id}` - Get claim details
- `GET /claims` - List all claims
- `GET /assessments/{id}` - Get risk assessment
- `GET /graphs/subgraph/{id}` - Get propagation subgraph
- `WS /ws/assessments` - Real-time WebSocket stream

## Project Structure

```
misinfo-detector/
├── config/           # Configuration files
├── src/
│   ├── api/          # FastAPI endpoints
│   ├── dashboard/    # Streamlit dashboard
│   ├── graph/        # Propagation graph builder
│   ├── ingestion/    # Data ingestion sources
│   ├── models/       # ML models (GNN, scoring)
│   ├── preprocessing/# Text encoding
│   ├── storage/      # Persistence layer
│   └── streaming/    # Event queue
├── tests/            # Unit and integration tests
├── notebooks/        # Jupyter analysis
├── data/             # Sample data
└── scripts/          # Utility scripts
```

## Configuration

See `config/config.yaml` for all settings. Environment variables can override config values:

- `MISINFO_CONFIG` - Path to config file
- `NEO4J_PASSWORD` - Neo4j password
- `REDIS_PASSWORD` - Redis password

## Development

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Streaming | Apache Kafka + Flink |
| NLP | HuggingFace Transformers |
| GNN | PyTorch Geometric |
| Graph DB | Neo4j |
| API | FastAPI |
| Dashboard | Streamlit |
| Explainability | SHAP |
