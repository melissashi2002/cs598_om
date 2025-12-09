# CS598 OM - PLEX Implementation

This repository contains implementations and tools for **PLEX (Probabilistic Local Explanation)**, a lightweight and efficient alternative to LIME for generating word-level importance scores in text classification models.

## Main Components

### 📁 `plex_training/`

The main implementation directory for training and evaluating PLEX models. This folder contains:

- **Training scripts**: Extract embeddings, generate LIME labels, and train PLEX models
- **Evaluation tools**: Compare PLEX vs LIME performance with stress tests and overlap metrics
- **Demo and visualization**: Interactive examples showing PLEX word importance scores

**Key Features:**
- Trains a lightweight SeismicNet to predict word importance from embeddings
- ~100x faster than LIME at inference time
- Maintains comparable faithfulness to LIME explanations

See [`plex_training/README.md`](plex_training/README.md) for detailed documentation.

### 📁 `Interview-Plex/`

A user study tool for measuring the impact of explanation methods on human judgment efficiency. This tool tests how long it takes humans to make judgments (e.g., toxicity classification) under different conditions:

- **Group 1 (Sentences 1-15)**: Original sentences without any explanation annotations
- **Group 2 (Sentences 16-30)**: Sentences with LIME word importance annotations
- **Group 3 (Sentences 31-45)**: Sentences with PLEX word importance annotations

**Contents:**
- `index.html`: Interactive HTML interface that tracks time spent in each group
- `generate_data.py`: Script to generate sample data with PLEX and LIME scores
- `data.csv`: Sample dataset with sentences and corresponding importance scores

**Usage:**
1. Prepare your data with PLEX and LIME scores
2. Open `index.html` in a web browser
3. Navigate through sentences and make judgments (e.g., Toxic/Not Toxic)
4. The tool automatically tracks time spent in each group
5. Export results to CSV for analysis

This tool is designed for:
- User studies comparing explanation method effectiveness
- Measuring how explanation annotations affect human decision-making speed
- Evaluating the practical utility of PLEX vs LIME in human-in-the-loop scenarios

## Quick Start

1. **Train a PLEX model**: See `plex_training/README.md` for step-by-step instructions
2. **Visualize results**: Use the `Interview-Plex/` tool to create interactive visualizations

## Requirements

```bash
pip install torch transformers pandas numpy tqdm scipy regex lime matplotlib
```

## License

See individual files for license information.

