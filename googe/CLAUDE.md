# CLAUDE.md - Misinformation Graph Detector

## Project Overview

Real-time misinformation detection system using propagation graphs, GNN, and temporal prediction. The system models how false claims spread through social networks and predicts virality before peaks occur.

## Architecture

The system has 6 layers:
1. **Ingestion** - Stream posts from social APIs (relational data, not just content)
2. **Kafka + Flink** - Handle millions of events/sec, deduplicate claims, build propagation snapshots
3. **NLP Encoder** - DeBERTa/RoBERTa for claim embeddings + stance detection
4. **Propagation GNN** - GraphSAGE/GAT learning structural patterns of viral misinformation
5. **Temporal Predictor** - LSTM on graph snapshots predicting virality
6. **Output** - Risk scores, API, SHAP explainability, dashboard

## Key Abstractions

### Core Types (`src/types.py`)
- `Claim` - misinformation claim with metadata
- `ClaimNode` - node in propagation graph
- `PropagationEdge` - edge representing share/retweet/reply
- `GraphSnapshot` - time-windowed snapshot of propagation graph
- `RiskAssessment` - final risk output with explanation

### Interfaces
- `IngestionSource` - abstract for data sources (stub for Twitter/Reddit/Facebook APIs)
- `GraphStore` - abstract for graph storage (in-memory NetworkX, Neo4j stub)
- `PropagationModel` - abstract for ML models (GraphSAGE implemented)

## Configuration

All config via `config/config.yaml`. Key settings:
- `models.encoder.name` - HuggingFace model (default: microsoft/deberta-v3-base)
- `models.gnn.type` - GNN type (graphsage)
- `streaming.use_kafka` - false for MVP, true for production
- `storage.neo4j.*` - Neo4j connection settings

## Running the Project

### Docker (Recommended)
```bash
docker-compose -f docker-compose.dev.yml up
```

### Local
```bash
pip install -r requirements.txt
python scripts/seed_data.py
uvicorn src.api.main:app --reload
streamlit run src/dashboard/app.py
```

## API

- `POST /claims` - Submit claim, returns RiskAssessment
- `GET /claims/{id}` - Get claim with metrics
- `GET /assessments/{id}` - Get risk assessment
- `GET /graphs/subgraph/{id}?depth=3` - Get propagation subgraph
- `WS /ws/assessments` - Real-time risk stream

## Testing

```bash
pytest tests/unit/ -v
```

## Important Notes

- GNN model is stubbed with heuristic scoring for MVP
- Neo4j adapter is a stub (would require actual Neo4j connection)
- Kafka/Flink replaced with asyncio queues for MVP
- Full production would need: trained models, real social API credentials, GPU resources
