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

An interactive web-based visualization tool for comparing PLEX and LIME word importance scores.

**Contents:**
- `index.html`: Interactive HTML interface for visualizing sentence-level word importance
- `generate_data.py`: Script to generate sample data with PLEX and LIME scores
- `data.csv`: Sample dataset with sentences and corresponding importance scores

**Usage:**
1. Generate or prepare your data with PLEX and LIME scores
2. Open `index.html` in a web browser
3. Navigate through sentences to see side-by-side PLEX vs LIME visualizations

This tool is useful for:
- Demonstrating PLEX capabilities in interviews or presentations
- Visual comparison of PLEX and LIME explanations
- Interactive exploration of word importance scores

## Quick Start

1. **Train a PLEX model**: See `plex_training/README.md` for step-by-step instructions
2. **Visualize results**: Use the `Interview-Plex/` tool to create interactive visualizations

## Requirements

```bash
pip install torch transformers pandas numpy tqdm scipy regex lime matplotlib
```

## License

See individual files for license information.

