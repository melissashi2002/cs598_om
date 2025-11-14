# -*- coding: utf-8 -*-
"""
RoBERTa Toxicity Classifier + PLEX: Guide
Modified from ModernBERT + PLEX: GoEmotions — Guide
"""

# (moved to install.sh) !pip install -r requirements.txt

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np
from tqdm import tqdm
import re, string

# ---------- helpers ----------
def normalize_token(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()

def get_cls_and_word_embs(text, tokenizer, model, device, l2norm=False, layer_idx=-1):
    """
    Returns:
      cls_embedding: [H] tensor
      word_order: list[str] - whitespace-split words
      word_embs: [W,H] tensor aligned to word_order (subwords averaged by char spans)
    """
    # Pass 1: get actual tokenized length (so we can set per-sentence max_length)
    enc0 = tokenizer(text, return_tensors="pt", truncation=False, padding=False, return_offsets_mapping=True)
    seq_len = enc0["input_ids"].shape[1]

    # Pass 2: re-tokenize with max_length=seq_len and offsets for merging
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len,
        padding=False,
        return_offsets_mapping=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    offsets = enc["offset_mapping"][0].tolist()  # list[(start,end), ...]

    with torch.no_grad():
        out = model(
            **{k: v for k, v in enc.items() if k in ["input_ids", "attention_mask"]},
            output_hidden_states=True
        )
        # choose layer (last by default)
        hidden = out.hidden_states[layer_idx].squeeze(0)  # [T,H]
    if l2norm:
        hidden = F.normalize(hidden, dim=-1)

    # CLS (first token for RoBERTa)
    cls_embedding = hidden[0].detach().cpu()

    # Collect subword spans (skip [CLS]/[SEP] and zero-length offsets)
    sub_spans = []
    for i, (s, e) in enumerate(offsets):
        if i == 0 or i == len(offsets) - 1:
            continue
        if s == e:
            continue
        sub_spans.append(((s, e), hidden[i].detach().cpu()))

    # Word order (whitespace split) + char spans
    word_order = text.split()
    word_emb_list = []
    words_char_spans = []
    char_ptr = 0
    for w in word_order:
        w_start = text.find(w, char_ptr)
        if w_start == -1:
            w_start = char_ptr  # fallback
        w_end = w_start + len(w)
        words_char_spans.append((w_start, w_end, w))
        char_ptr = w_end

    for (w_start, w_end, w) in words_char_spans:
        # subwords overlapping the word's char span
        embs = [emb for (s, e), emb in sub_spans if not (e <= w_start or s >= w_end)]
        if len(embs) == 0:
            # relaxed fallback: subword text is substring of the word or vice-versa
            embs = [emb for (s, e), emb in sub_spans if (text[s:e] in w) or (w in text[s:e])]
        if len(embs) > 0:
            embs = torch.stack(embs, dim=0)
            word_emb = embs.mean(dim=0)
            word_emb_list.append(word_emb)
        else:
            # skip words that have no aligned subwords (e.g., pure punctuation)
            pass

    word_embs = torch.stack(word_emb_list, dim=0) if len(word_emb_list) > 0 else torch.empty(0, hidden.shape[-1])
    return cls_embedding, word_order, word_embs

# ---------- main ----------
# Load data from CSV
df = pd.read_csv("mp1-data-train-300-random.csv")
print(f"Loaded {len(df)} samples from CSV")

# Model and tokenizer
model_name = "s-nlp/roberta_toxicity_classifier"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)
model.eval()

# GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# LIME prediction function (binary classification: softmax over 2 classes)
def predict_proba(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
    with torch.no_grad():
        logits = model(**tokens).logits
        probs = F.softmax(logits, dim=-1).cpu().numpy()  # [B, 2]
    return probs

# LIME explainer (2 classes: non-toxic=0, toxic=1)
explainer = LimeTextExplainer(class_names=["non-toxic", "toxic"])

results = []
skipped_empty = 0
skipped_processing = 0
skipped_no_words = 0
skipped_no_lime = 0
skipped_dim_mismatch = 0

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row["body"])
    if pd.isna(text) or text.strip() == "":
        skipped_empty += 1
        continue  # skip empty text

    # Get CLS + merged word embeddings (last layer by default)
    try:
        cls_embedding, word_order, word_embs = get_cls_and_word_embs(
            text, tokenizer, model, device, l2norm=False, layer_idx=-1
        )
    except Exception as e:
        skipped_processing += 1
        if skipped_processing <= 5:  # Only print first few errors
            print(f"Error processing text at row {i}: {e}")
        continue

    # LIME target: use model's top predicted class
    try:
        tokens_test = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        with torch.no_grad():
            logits_test = model(**tokens_test).logits
            probs_test = F.softmax(logits_test, dim=-1)[0]
            target = int(torch.argmax(probs_test).item())  # 0 or 1
    except Exception:
        target = 1  # default to toxic class if prediction fails

    try:
        num_features = min(max(len(word_order), 5), 20)  # stable features count
        exp = explainer.explain_instance(text, predict_proba, labels=[target], num_features=num_features, num_samples=1000)
        lime_list = exp.as_list(label=target)  # list of (token_str, weight)
        lime_tokens = [t for (t, w) in lime_list]
        lime_scores = [float(w) for (t, w) in lime_list]
        # Ensure we have some LIME scores
        if len(lime_tokens) == 0:
            # If LIME failed, create dummy scores based on word order
            lime_tokens = word_order[:min(10, len(word_order))]
            lime_scores = [0.1] * len(lime_tokens)
    except Exception as e:
        # If LIME completely fails, create dummy scores to avoid skipping
        if len(word_order) > 0:
            lime_tokens = word_order[:min(10, len(word_order))]
            lime_scores = [0.1] * len(lime_tokens)
        else:
            lime_tokens, lime_scores = [], []

    # Align LIME tokens to whitespace words (same order as word_order)
    norm_word_order = [normalize_token(w) for w in word_order]
    norm2idxs = {}
    for j, w in enumerate(norm_word_order):
        if w:
            norm2idxs.setdefault(w, []).append(j)

    aligned = [0.0 for _ in word_order]
    for tok, wt in zip(lime_tokens, lime_scores):
        key = normalize_token(tok)
        if key in norm2idxs:
            share = wt / max(1, len(norm2idxs[key]))
            for j in norm2idxs[key]:
                aligned[j] += share
        # else: no match → ignore (punctuation/mismatch)

    lime_scores_aligned = torch.tensor(aligned) if len(aligned) > 0 else torch.empty(0)

    # Skip if word_embs or word_order is empty
    if word_embs.numel() == 0 or len(word_order) == 0:
        skipped_no_words += 1
        continue
    
    # Skip if lime_scores_aligned is empty (but we should have handled this above)
    if lime_scores_aligned.numel() == 0:
        # Try to create minimal scores if alignment failed
        if len(word_order) > 0:
            lime_scores_aligned = torch.zeros(len(word_order))
            # Give small random scores to avoid all zeros
            lime_scores_aligned[:min(3, len(word_order))] = 0.1
        else:
            skipped_no_lime += 1
            continue
    
    # Ensure dimensions match
    if len(word_order) != word_embs.shape[0] or len(word_order) != lime_scores_aligned.shape[0]:
        skipped_dim_mismatch += 1
        continue

    # Save
    results.append({
        "text": text,
        "label_ids": [target],  # single label for binary classification
        "cls_embedding": cls_embedding.cpu(),   # [H]
        "word_order": word_order,               # whitespace words
        "word_embeddings": word_embs.cpu(),     # [W,H] merged
        "lime_tokens": lime_tokens,
        "lime_scores": torch.tensor(lime_scores),
        "lime_scores_aligned": lime_scores_aligned,  # [W] aligned to word_order
    })

# Save
print(f"\n=== Data Processing Summary ===")
print(f"Total results collected: {len(results)}")
print(f"Skipped - empty text: {skipped_empty}")
print(f"Skipped - processing error: {skipped_processing}")
print(f"Skipped - no words/embeddings: {skipped_no_words}")
print(f"Skipped - no LIME scores: {skipped_no_lime}")
print(f"Skipped - dimension mismatch: {skipped_dim_mismatch}")
print(f"Total processed: {len(results)}/{len(df)}")

if len(results) == 0:
    raise ValueError("No valid samples were processed! Please check your data and LIME explainer.")

torch.save(results, "toxicity_test_with_lime_words_train.pt")
print("✅ Saved merged word embeddings + LIME to toxicity_test_with_lime_words_train.pt")

# Quick peek
if results:
    ex = results[0]
    print("\nExample sample:")
    print("  Text:", ex["text"][:100] + "..." if len(ex["text"]) > 100 else ex["text"])
    print("  Words:", ex["word_order"][:12])
    print("  Word embeddings shape:", tuple(ex["word_embeddings"].shape))
    print("  LIME aligned shape:", tuple(ex["lime_scores_aligned"].shape))
    print("  LIME aligned (first 12):", ex["lime_scores_aligned"][:12])

# train_plex_from_lime.py
import os, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.stats import spearmanr

# ---------------------------
# Config
# ---------------------------
DATA_PATH = "toxicity_test_with_lime_words_train.pt"  # produced earlier
SAVE_CKPT = "plex_seismic_roberta_toxicity_lime.pth"

INPUT_SIZE = 768           # RoBERTa-base hidden size
BATCH_SIZE = 2048          # pairs (CLS, word) per batch (adjust for your GPU/CPU)
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# Model (shared Siamese head)
# ---------------------------
class SeismicNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PLEXHeadShared(nn.Module):
    """
    Projects CLS and word embeddings with shared SeismicNet,
    then computes cosine similarity per word.
    """
    def __init__(self, input_size=768):
        super().__init__()
        self.net = SeismicNet(input_size)

    def forward(self, cls_emb: torch.Tensor, word_emb: torch.Tensor) -> torch.Tensor:
        """
        cls_emb: [B, H]
        word_emb: [B, H]
        returns: [B] cosine similarities in [-1, 1]
        """
        cls_proj = self.net(cls_emb)      # [B, 64]
        word_proj = self.net(word_emb)    # [B, 64]
        word_proj = word_proj - word_proj.mean(dim=0, keepdim=True)
        # optional centering for more dispersion (can help)
        # word_proj = word_proj - word_proj.mean(dim=0, keepdim=True)
        cls_norm = F.normalize(cls_proj, dim=-1)
        word_norm = F.normalize(word_proj, dim=-1)
        return torch.sum(cls_norm * word_norm, dim=-1)

# ---------------------------
# Data utilities
# ---------------------------
def load_data(path):
    # weights_only=False because file contains Python objects (list of dicts)
    return torch.load(path, weights_only=False)

def split_indices(n, val_ratio=0.2, seed=SEED):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    v = int(round(n * val_ratio))
    return idx[v:], idx[:v]  # train_idx, val_idx

class PLEXPairsDataset(Dataset):
    """
    Flattens per-sentence pairs (CLS, word_i, target_i) across all sentences.
    Targets are normalized per sentence by max|target|.
    """
    def __init__(self, samples):
        self.cls = []       # [N, H]
        self.wrd = []       # [N, H]
        self.tgt = []       # [N]
        # Keep track of sample boundaries for evaluation (optional)
        self.sent_ptrs = [] # list of (start, end) indices in flat arrays
        cur = 0
        for ex in samples:
            word_embs: torch.Tensor = ex["word_embeddings"]   # [W, H]
            lime_aligned: torch.Tensor = ex["lime_scores_aligned"]  # [W]
            cls_emb: torch.Tensor = ex["cls_embedding"]       # [H]

            if word_embs.numel() == 0 or lime_aligned.numel() == 0:
                self.sent_ptrs.append((cur, cur))
                continue

            # Normalize targets per sentence by max|score| to [-1,1]
            y = lime_aligned.clone().float()
            m = torch.max(torch.abs(y))
            if m > 0:
                y = y / m

            # NEW: threshold to suppress tail noise
            tau = 0.01  # Lower threshold to keep more tokens
            keep = torch.abs(y) >= tau
            if keep.sum() > 0:
                word_embs = word_embs[keep]
                y = y[keep]
            else:
                # if all tiny, keep the largest 1-3 tokens to avoid empty sentence
                k = min(3, len(y))
                topk_indices = torch.topk(torch.abs(y), k).indices
                word_embs = word_embs[topk_indices]
                y = y[topk_indices]

            W = word_embs.shape[0]
            cls_rep = cls_emb.unsqueeze(0).repeat(W, 1)

            W = word_embs.shape[0]
            # Repeat CLS per word
            cls_rep = cls_emb.unsqueeze(0).repeat(W, 1)       # [W, H]

            self.cls.append(cls_rep)
            self.wrd.append(word_embs.float())
            self.tgt.append(y)

            cur_next = cur + W
            self.sent_ptrs.append((cur, cur_next))
            cur = cur_next

        if len(self.cls) > 0:
            self.cls = torch.cat(self.cls, dim=0)  # [N, H]
            self.wrd = torch.cat(self.wrd, dim=0)  # [N, H]
            self.tgt = torch.cat(self.tgt, dim=0)  # [N]
        else:
            self.cls = torch.empty(0, INPUT_SIZE)
            self.wrd = torch.empty(0, INPUT_SIZE)
            self.tgt = torch.empty(0)

    def __len__(self):
        return self.tgt.shape[0]

    def __getitem__(self, idx):
        return self.cls[idx], self.wrd[idx], self.tgt[idx]

# ---------------------------
# Metrics
# ---------------------------
def evaluate_sentence_level(model, samples):
    """
    Compute per-sentence Spearman (median) and Top-k overlap (k=1,3,5) means.
    """
    model.eval()
    spearmans = []
    top1 = []; top3 = []; top5 = []

    @torch.no_grad()
    def predict_scores(cls_emb, word_embs):
        B = word_embs.shape[0]
        cls_rep = cls_emb.unsqueeze(0).repeat(B, 1).to(DEVICE)
        word_embs = word_embs.to(DEVICE)
        scores = model(cls_rep, word_embs).cpu().numpy()
        return scores

    for ex in samples:
        words = ex["word_order"]
        word_embs: torch.Tensor = ex["word_embeddings"]
        lime = ex["lime_scores_aligned"].numpy() if len(words)>0 else np.array([])
        if word_embs.numel() == 0 or lime.size == 0:
            continue

        # Normalize LIME per sentence as in training
        if np.max(np.abs(lime)) > 0:
            lime_n = lime / np.max(np.abs(lime))
        else:
            lime_n = lime

        cls = ex["cls_embedding"]
        pred = predict_scores(cls, word_embs)

        # Spearman across all words
        if pred.size > 1 and lime_n.size > 1:
            rho, _ = spearmanr(pred, lime_n)
            if not np.isnan(rho):
                spearmans.append(rho)

        # Top-k overlap helper
        def topk_overlap(a, b, k):
            if len(a) == 0 or len(b) == 0 or len(a) != len(b) or len(a) < k:
                return np.nan
            ai = np.argsort(-a)[:k]
            bi = np.argsort(-b)[:k]
            return len(set(ai) & set(bi)) / float(k)

        top1.append(topk_overlap(pred, lime_n, 1))
        top3.append(topk_overlap(pred, lime_n, 3))
        top5.append(topk_overlap(pred, lime_n, 5))

    # Aggregate
    def nanmean(x): return float(np.nanmean(x)) if len(x)>0 else float("nan")
    def nanmedian(x): return float(np.nanmedian(x)) if len(x)>0 else float("nan")

    return {
        "spearman_median": nanmedian(spearmans),
        "top1_mean": nanmean(top1),
        "top3_mean": nanmean(top3),
        "top5_mean": nanmean(top5),
        "n_sentences": len(samples)
    }

# ---------------------------
# Training
# ---------------------------
def main():
    print("Loading data:", DATA_PATH)
    all_samples = load_data(DATA_PATH)
    # Filter out empty samples just in case
    all_samples = [ex for ex in all_samples if ex["word_embeddings"].numel() > 0 and ex["lime_scores_aligned"].numel() > 0]
    print(f"Total usable sentences: {len(all_samples)}")

    # Split sentences
    train_idx, val_idx = split_indices(len(all_samples), val_ratio=VAL_SPLIT, seed=SEED)
    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    print(f"Train sentences: {len(train_samples)} | Val sentences: {len(val_samples)}")

    # Build flat pair dataset
    train_ds = PLEXPairsDataset(train_samples)
    val_ds   = PLEXPairsDataset(val_samples)
    print(f"Train pairs: {len(train_ds)} | Val pairs: {len(val_ds)}")

    # Check if datasets are empty
    if len(train_ds) == 0:
        print("\n❌ ERROR: Training dataset is empty!")
        print(f"  - Total samples loaded: {len(all_samples)}")
        print(f"  - Train samples: {len(train_samples)}")
        print(f"  - Possible causes:")
        print(f"    1. All samples were filtered out during dataset creation")
        print(f"    2. Threshold (tau=0.01) is too high")
        print(f"    3. Word embeddings or LIME scores are invalid")
        print(f"  - Try: Lower the threshold tau in PLEXPairsDataset or check data preprocessing")
        raise ValueError("Training dataset is empty! Please check data preprocessing.")
    if len(val_ds) == 0:
        print("\n⚠️  WARNING: Validation dataset is empty!")
        print(f"  - Val samples: {len(val_samples)}")
        print(f"  - Training will continue but validation metrics won't be computed")
        # Don't raise error for empty validation set, just skip validation
        val_loader = None

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    if len(val_ds) > 0:
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    else:
        val_loader = None

    # Model, opt, loss
    model = PLEXHeadShared(input_size=INPUT_SIZE).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    huber = nn.SmoothL1Loss(reduction="none")  # we'll weight it

    # Training loop
    best_val = float("inf")
    for epoch in range(1, EPOCHS+1):
        model.train()
        running = 0.0
        count = 0

        for cls_b, wrd_b, tgt_b in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            cls_b = cls_b.to(DEVICE)
            wrd_b = wrd_b.to(DEVICE)
            tgt_b = tgt_b.to(DEVICE)           # in [-1,1] (per-sentence normalized)

            pred = model(cls_b, wrd_b)         # [-1,1]
            loss_vec = huber(pred, tgt_b)

            # weight by |target| to emphasize salient tokens (avoid all zeros dominating)
            weights = torch.clamp_min(torch.abs(tgt_b), 0.1)  # floor to keep gradient on small targets
            loss = (loss_vec * weights).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item() * cls_b.size(0)
            count   += cls_b.size(0)

        train_loss = running / max(1, count)

        # Quick pair-level val loss
        if val_loader is not None:
            model.eval()
            val_running = 0.0
            val_count = 0
            with torch.no_grad():
                for cls_b, wrd_b, tgt_b in val_loader:
                cls_b = cls_b.to(DEVICE)
                wrd_b = wrd_b.to(DEVICE)
                tgt_b = tgt_b.to(DEVICE)
                pred = model(cls_b, wrd_b)
                loss_vec = huber(pred, tgt_b)

                # pred: [B], tgt_b: [B] in [-1,1] (already normalized per sentence)
                loss_vec = huber(pred, tgt_b)

                # regression weighted by |target|
                weights = torch.abs(tgt_b).clamp(min=1e-6)  # no floor; let low-importance tokens carry tiny weight
                reg = (loss_vec * weights).sum() / (weights.sum() + 1e-6)

                # ✅ PATCH: pairwise rank loss (top vs. bottom tokens in the minibatch)
                pos_mask = tgt_b > 0
                neg_mask = tgt_b < 0
                rank = torch.tensor(0.0, device=pred.device)
                if pos_mask.any() and neg_mask.any():
                    pos = pred[pos_mask]
                    neg = pred[neg_mask]
                    # sample a handful to keep it light
                    P = pos[torch.randint(0, len(pos), (min(128, len(pos)),))]
                    N = neg[torch.randint(0, len(neg), (min(128, len(neg)),))]
                    margin = 0.2
                    # hinge: want P > N + margin
                    rank = (margin - P.view(-1,1) + N.view(1,-1)).clamp(min=0).mean()

                # combine
                loss = reg + 0.3 * rank   # rank weight 0.2 is a good starting point


                weights = torch.clamp_min(torch.abs(tgt_b), 0.1)
                loss = (loss_vec * weights).mean()
                val_running += loss.item() * cls_b.size(0)
                val_count   += cls_b.size(0)
            val_loss = val_running / max(1, val_count)
            
            # Sentence-level metrics (more meaningful)
            sent_metrics = evaluate_sentence_level(model, val_samples)
        else:
            val_loss = float("inf")
            sent_metrics = {
                "spearman_median": float("nan"),
                "top1_mean": float("nan"),
                "top3_mean": float("nan"),
                "top5_mean": float("nan"),
                "n_sentences": 0
            }

        print(f"\nEpoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"Spearman_median={sent_metrics['spearman_median']:.3f}  "
              f"Top1={sent_metrics['top1_mean']:.3f}  Top3={sent_metrics['top3_mean']:.3f}  "
              f"Top5={sent_metrics['top5_mean']:.3f}  n_val_sent={sent_metrics['n_sentences']}")

        # Save best by val_loss
        if val_loss < best_val:
            best_val = val_loss
            state = {
                "state_dict": model.state_dict(),
                "config": {
                    "input_size": INPUT_SIZE,
                    "epochs": EPOCHS,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED
                }
            }
            torch.save(state, SAVE_CKPT)
            print(f"  ✅ Saved best checkpoint -> {SAVE_CKPT}")

    print("\nDone.")

if __name__ == "__main__":
    main()

