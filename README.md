# Parallel Memory System

A psychologically-informed AI system that tracks how users' memories evolve over time, detects emotional and narrative distortions, and generates alternate life timeline simulations.

## Core Concept

Humans don't store memories like recordings — every recall changes details, emotions shift, and missing parts get reconstructed. **Parallel Memory** combines memory distortion psychology with AI-generated alternate timeline simulations into one unified system.

## Features

- **Memory Embedding & Drift Detection** — Detects semantic and emotional changes between original memory and recall
- **Neural Network-Powered Confidence Scoring** — Estimates reliability of memory recall over time
- **Emotion Classification** — Categorizes emotional tone (regret, nostalgia, sadness, joy, etc.)
- **Alternate Timeline Generation** — Simulates counterfactual life branches from decision points
- **User Feedback Loop** — Collects corrections to continuously improve models
- **Auto-Retraining** — LoRA fine-tuning on aggregated negative feedback samples
- **Colab-Ready** — Runs entirely in Google Colab with GPU acceleration
- **GitHub-Integrated** — Auto-commits model artifacts and results to GitHub

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | FastAPI + REST API |
| Backend | Python 3.10+ |
| Storage | SQLite (per-user + global) |
| Embeddings | SentenceTransformer (all-MiniLM-L6-v2) |
| Neural Networks | PyTorch (MemoryDriftNN, ConfidenceScorer, EmotionClassifier) |
| LLM | TinyLlama with LoRA fine-tuning |
| Orchestration | LangGraph (optional) |
| ML Tooling | Hugging Face Transformers, PEFT, Accelerate |
| Monitoring | Weights & Biases (optional) |
| Execution | Google Colab (primary) |

## Installation

### Local Development

```bash
git clone https://github.com/Aditya2005-cloud/PARALLER-MEMORY.git
cd PARALLER_MEMORY

# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/users data/global logs models checkpoints
```

### Google Colab

1. Open [Google Colab](https://colab.research.google.com)
2. Go to `File` → `Open notebook` → `GitHub`
3. Enter: `Aditya2005-cloud/PARALLER-MEMORY`
4. Select `colab_parallel_memory.ipynb`
5. Run cells sequentially

## Quick Start

### 1. Local API Server

```bash
uvicorn parallel_memory.api:app --reload --port 8000
```

Visit http://localhost:8000/docs for API documentation.

### 2. Submit a Memory

```bash
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "memory_text": "In 2022 I rejected a music scholarship for engineering",
    "emotion": "uncertain",
    "confidence": 0.65
  }'
```

### 3. Recall & Detect Drift

```bash
curl -X POST http://localhost:8000/recalls \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "memory_id": "<returned_memory_id>",
    "recall_text": "I was forced to reject music because of family pressure",
    "emotion": "regret",
    "confidence": 0.4
  }'
```

**Response includes:**
- Semantic drift score (0-1): how much the meaning changed
- Emotion shift score (0-1): emotional tone change
- Omitted/added keywords: what changed in the telling
- 3 alternate timeline branches

### 4. Submit Feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "memory_id": "<memory_id>",
    "response_id": "<response_id>",
    "rating": 0.2,
    "correction": "The drift analysis was too aggressive",
    "notes": "Model underestimated emotional continuity"
  }'
```

## Architecture

### Database Schema

**Per-user database** (`data/users/{user_id}.db`):
- `memories` — original memory submissions
- `embeddings` — cached 384-dim vectors
- `recalls` — re-tellings with detected drift
- `timelines` — alternate scenario branches
- `feedback` — user corrections
- `exceptions` — errors for debugging

**Global database** (`data/global.db`):
- `global_feedback` — anonymized, aggregated feedback
- `retrain_log` — model training history

### Pipeline Flow

```
User Input
    ↓
Embedding (SentenceTransformer)
    ↓
Drift Detection NN + Emotion Classifier
    ↓
Confidence Scoring NN
    ↓
Alternate Timeline Generation
    ↓
Database Storage
    ↓
User Feedback Loop
    ↓
Auto-Retrain (when threshold hit)
```

## Neural Network Models

### MemoryDriftNN
- **Input**: Previous embedding (384D) + current embedding (384D)
- **Output**: Semantic drift score, emotion drift score (both 0-1)
- **Architecture**: LSTM + Attention + Dense layers
- **Training**: Supervised on user-labeled drift annotations

### ConfidenceScorer
- **Input**: Memory embedding + recall count + days old + text length
- **Output**: Confidence score (0-1, higher = more reliable)
- **Architecture**: Multi-layer perceptron
- **Training**: User feedback ratings map to confidence labels

### EmotionClassifier
- **Input**: Text embedding (384D)
- **Output**: Probability distribution over 8 emotions
- **Classes**: joy, sadness, regret, acceptance, nostalgia, anger, shame, pride
- **Architecture**: Dense network with softmax
- **Training**: User emotion labels from recalled memories

## Data Migration

Existing JSON files are automatically migrated to SQLite:

```python
from parallel_memory.migration import MigrationManager

migration = MigrationManager()
results = migration.run_full_migration(users=["user_1", "user_2"])
migration.archive_json_files()
```

## Retraining Pipeline

Models retrain automatically when:
- ≥20 negative feedback samples accumulated
- ≥30% negative ratio in recent 50 responses
- ≥10 unresolved exceptions
- 7+ days since last retraining

Check retrain status:

```bash
curl http://localhost:8000/global/retrain-check
```

## Error Handling

All failures are caught and logged:

```python
from parallel_memory.exceptions import (
    MemoryError,
    EmbeddingError,
    ModelInferenceError,
    DatabaseError,
    UserValidationError
)
```

Fallbacks:
- Embedding failures → keyword similarity fallback
- Model load failures → fresh model initialization
- GPU OOM → automatic CPU fallback
- Database locks → exponential backoff + retry

## Logging

Logs written to `logs/parallel_memory_*.log`:

```python
import logging
logger = logging.getLogger("parallel_memory")
logger.info("Your message here")
```

## Testing

Run tests locally:

```bash
pytest tests/ -v --cov=parallel_memory
```

## Colab Deployment

1. **Mount GitHub token** (for auto-commits):
   ```python
   from google.colab import userdata
   github_token = userdata.get('GITHUB_TOKEN')
   os.environ['GITHUB_TOKEN'] = github_token
   ```

2. **Run notebook cells** in order (setup → training → test → commit)

3. **Check GitHub** for committed model weights and results.json

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| POST | `/memories` | Store new memory |
| POST | `/recalls` | Process recall + drift detection |
| POST | `/feedback` | Log user feedback |
| GET | `/user/{user_id}/memories` | List user's memories |
| GET | `/user/{user_id}/feedback-stats` | Get user feedback stats |
| GET | `/global/summary` | Aggregate feedback stats |
| POST | `/global/retrain-check` | Check if retraining triggered |
| GET | `/errors` | Recent error logs |

## Performance

On T4 GPU (Colab):
- Memory embedding: ~50ms per text
- Drift detection: ~200ms per recall
- Timeline generation: ~300ms per memory
- Full pipeline (recall → timeline): ~1s

## Safety & Ethics

- All alternate timelines labeled "AI Simulation"
- Confidence scores are probabilistic, never claim truth
- No attempts to suppress or maximize particular emotions
- User retains full control over data
- No external API calls (runs fully local/Colab)

## Future Roadmap

- [ ] Fine-tune TinyLlama on user corpus (curriculum learning)
- [ ] Add regret trajectory visualization (Matplotlib/Plotly)
- [ ] Implement RAG for memory context retrieval
- [ ] Multi-user collaboration features
- [ ] React frontend with branching timeline UI
- [ ] Mobile app (React Native)
- [ ] Publish research paper on memory distortion patterns

## Contributing

We welcome contributions! Areas of interest:
- Improved memory drift detection algorithms
- Better timeline generation heuristics
- Expanded emotion classification
- Frontend UI/UX
- Testing & documentation

## License

This project is open-source. See LICENSE file.

## Authors

- Aditya2005-cloud

## Support

Questions? Open an issue on GitHub or check the API docs at `http://localhost:8000/docs`

---

**Remember**: This system is designed to help you *understand* your memories better, not to claim objective truth about them. Memories are reconstructed every time we recall them — that's not a bug, it's a feature of human cognition.
