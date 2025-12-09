# PLEX Training Implementation

This directory contains the main implementation for training and evaluating **PLEX (Probabilistic Local Explanation)** models. PLEX is a lightweight, efficient alternative to LIME for generating word-level importance scores in text classification tasks.

## Overview

PLEX learns to predict word importance scores by training a small neural network (SeismicNet) to match LIME explanations. The key advantage is that PLEX can generate explanations much faster than LIME at inference time, while maintaining comparable faithfulness to the underlying model.

### Key Features

- **Efficient Inference**: Once trained, PLEX generates word importance scores in a single forward pass, avoiding the expensive sampling required by LIME
- **Faithful Explanations**: Trained to match LIME scores, maintaining similar faithfulness metrics
- **Lightweight Architecture**: Uses a small shared Siamese network (SeismicNet) with only ~100K parameters
- **Flexible**: Works with any transformer-based classification model (RoBERTa, ModernBERT, etc.)

## Architecture

PLEX consists of two main components:

1. **SeismicNet**: A shared projection network that maps embeddings to a lower-dimensional space
   - Input: 768-dimensional embeddings (from base transformer models)
   - Architecture: `768 → 128 → 64` with ReLU activations and dropout
   
2. **PLEXHeadShared**: Computes cosine similarity between projected CLS and word embeddings
   - Projects both CLS and word embeddings through the shared SeismicNet
   - Returns cosine similarity scores in [-1, 1] range

## Files

### Core Scripts

- **`00_roberta_mp1data_plex_guide.py`**: Main training script
  - Extracts CLS and word-level embeddings from a fine-tuned classifier
  - Generates LIME explanations for training data
  - Trains the PLEX model to predict LIME scores
  - Saves trained model checkpoint

- **`01_roberta_mp1data_interpreting.py`**: Evaluation script
  - Loads trained PLEX model
  - Performs stress tests (probability drop when removing top-k words)
  - Computes overlap metrics (Jaccard similarity) between PLEX and LIME
  - Reports faithfulness metrics with confidence intervals

- **`02_demo.py`**: Interactive demo and visualization
  - Demonstrates how to use trained PLEX model on custom text
  - Generates visualizations comparing PLEX vs LIME word importance
  - Saves comparison plots to `plex_visualizations/` directory

- **`PLEX_Training_Using_Modernbert_base_go_emotions.ipynb`**: Jupyter notebook version
  - Alternative implementation for GoEmotions dataset
  - Includes step-by-step cells for data processing and training

### Data Files

- **`mp1-data-train-300-random.csv`**: Training data (300 samples)
- **`mp1data_with_lime_words.pt`**: Preprocessed data with embeddings and LIME scores
- **`plex_seismic_modernbert_lime.pth`**: Trained PLEX model checkpoint

### Visualization

- **`visualization/`**: Directory containing example visualization outputs
  - Comparison plots showing PLEX vs LIME word importance scores

## Installation

### Requirements

```bash
pip install torch transformers pandas numpy tqdm scipy regex lime matplotlib
```

Or use the requirements file from the parent directory:

```bash
pip install -r ../Officialtraining/requirements.txt
```

## Usage

### Step 1: Prepare Data and Extract Embeddings

First, ensure you have:
1. A fine-tuned classification model (e.g., RoBERTa fine-tuned on your task)
2. Training data in CSV format with text and labels

Modify `00_roberta_mp1data_plex_guide.py` to point to your data and model:

```python
CSV_PATH = "your_data.csv"
MODEL_DIR = "path/to/your/finetuned/model"
LIME_OUTPUT_PATH = "your_data_with_lime_words.pt"
```

Run the script to extract embeddings and generate LIME labels:

```bash
python 00_roberta_mp1data_plex_guide.py
```

This will:
- Load your data and fine-tuned model
- Extract CLS and word-level embeddings
- Generate LIME explanations for each sample
- Save everything to a `.pt` file

### Step 2: Train PLEX Model

The training code is included in `00_roberta_mp1data_plex_guide.py` (after the data extraction section). The script will automatically:

1. Load the preprocessed data with LIME scores
2. Split into train/validation sets
3. Train PLEX model using weighted Huber loss
4. Evaluate on validation set using:
   - Spearman correlation (median)
   - Top-k overlap (k=1, 3, 5)
5. Save the best checkpoint

Training configuration (can be modified in the script):

```python
BATCH_SIZE = 2048      # Pairs per batch
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
```

### Step 3: Evaluate Model

Run the evaluation script to assess PLEX performance:

```bash
python 01_roberta_mp1data_interpreting.py
```

This will:
- Load the trained PLEX model
- Compare PLEX vs LIME on test data
- Perform stress tests (removing top-k words and measuring probability drop)
- Report overlap statistics (Jaccard similarity)

Expected output:
```
Top-k removal Δprob (mean [95% CI]):
  LIME: 0.2390  [0.1878, 0.2902]
  PLEX: 0.2314  [0.1770, 0.2859]

Top-k 词重叠度分析 (LIME vs PLEX)
  Jaccard 相似度: 0.4523  [0.4123, 0.4923]
```

### Step 4: Demo and Visualization

Run the demo script to see PLEX in action:

```bash
python 02_demo.py
```

This will:
- Load the trained PLEX model
- Process custom text examples
- Generate visualizations comparing PLEX vs LIME
- Save plots to `plex_visualizations/` directory

## Model Details

### Training Objective

PLEX is trained to minimize a weighted Huber loss between predicted scores and normalized LIME scores:

```python
loss = Huber(pred, target) * weights
weights = clamp_min(|target|, 0.1)  # Emphasize salient tokens
```

Targets are normalized per sentence by `max(|LIME_scores|)` to [-1, 1] range.

### Data Processing

1. **Word Embedding Extraction**: 
   - Subword tokens are merged to word-level by averaging embeddings of subwords that overlap with each word's character span
   - Handles tokenization mismatches between LIME (word-level) and transformer (subword-level)

2. **LIME Alignment**:
   - LIME tokens are normalized and aligned to whitespace-split words
   - Scores are distributed across multiple word occurrences

3. **Thresholding**:
   - Words with very small LIME scores (|score| < 0.05) are filtered out during training
   - Helps focus on salient tokens and reduce noise

### Evaluation Metrics

1. **Spearman Correlation**: Rank correlation between PLEX and LIME scores (median across sentences)
2. **Top-k Overlap**: Fraction of words that appear in both PLEX and LIME top-k lists
3. **Probability Drop (Δprob)**: Drop in model's predicted probability when removing top-k words identified by PLEX/LIME
   - Higher drop indicates more faithful explanations
   - Comparable drops between PLEX and LIME indicate similar faithfulness

## Configuration

### Model Paths

Update these paths in each script to match your setup:

```python
MODEL_DIR = "askhistorians"  # Path to fine-tuned classifier
PLEX_CKPT = "askhistorians_plex_seismic_modernbert_lime.pth"  # PLEX checkpoint
DATA_PATH = "askhistorians_softmax_data_with_lime_words.pt"  # Preprocessed data
```

### Task-Specific Settings

For **binary classification** (e.g., toxicity detection):
- Uses `softmax` for probability prediction (2 classes)
- Target label is 0 or 1

For **multi-label classification** (e.g., GoEmotions):
- Uses `sigmoid` for probability prediction (28 classes)
- Target label is the top predicted class index

Modify the `predict_proba` function in each script accordingly.

## Output Format

### Saved Data (`.pt` file)

Each sample contains:
```python
{
    "text": str,                    # Original text
    "label_ids": List[int],         # Ground truth labels
    "cls_embedding": Tensor[H],     # CLS token embedding
    "word_order": List[str],        # Whitespace-split words
    "word_embeddings": Tensor[W,H], # Word-level embeddings
    "lime_tokens": List[str],       # LIME token strings
    "lime_scores": Tensor[L],       # LIME scores (unaligned)
    "lime_scores_aligned": Tensor[W] # LIME scores aligned to word_order
}
```

### Model Checkpoint (`.pth` file)

```python
{
    "state_dict": {...},            # Model weights
    "config": {
        "input_size": 768,
        "epochs": 50,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "seed": 42
    }
}
```

