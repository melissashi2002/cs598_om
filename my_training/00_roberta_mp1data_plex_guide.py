# -*- coding: utf-8 -*-
"""
Exported from: PLEX_Training_Using_Modernbert_base_go_emotions.ipynb
Section: ModernBERT + PLEX: GoEmotions — Guide
Export time: 2025-11-08T20:37:11
Notes:
  - IPython magics (%) are commented out.
  - Shell commands (!) were redirected to install.sh or commented.
"""


# (moved to install.sh) !pip install -r requirements.txt

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np
from tqdm import tqdm
import re, string

CSV_PATH = "mp1-data-train-300-random.csv"
LIME_OUTPUT_PATH = "mp1data_with_lime_words.pt"

# ---------- helpers ----------
def normalize_token(s: str) -> str:
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()

def parse_label_ids(raw_label):
    if raw_label is None:
        return []
    if isinstance(raw_label, bool):
        return [1 if raw_label else 0]
    if isinstance(raw_label, (int, np.integer)):
        return [int(raw_label)]
    if isinstance(raw_label, (float, np.floating)):
        if np.isnan(raw_label):
            return []
        return [int(raw_label)]
    if isinstance(raw_label, str):
        text = raw_label.strip()
        if not text:
            return []
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return [1 if lowered == "true" else 0]
        return [int(x) for x in text.split(",") if x.strip().isdigit()]
    return []

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

    # CLS
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
# Load mp1 CSV directly (no merge)
df = pd.read_csv(CSV_PATH)
if "body" not in df.columns:
    raise ValueError(f"'body' column missing in {CSV_PATH}")
df["text"] = df["body"].fillna("").astype(str)
df["labels"] = df.get("removed", np.nan)
df = df.reset_index(drop=True)

# Model and tokenizer
model_name = "s-nlp/roberta_toxicity_classifier"
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)
model.eval()

# GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
batch = tokenizer.encode("You are amazing!", return_tensors="pt").to(device)
num_labels = model.config.num_labels

# LIME prediction function (multi-label: sigmoid)
def predict_proba(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
    with torch.no_grad():
        logits = model(**tokens).logits
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs

# LIME explainer
explainer = LimeTextExplainer(class_names=[str(i) for i in range(num_labels)])

results = []
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row["text"])
    label_ids = parse_label_ids(row.get("labels"))
    if len(label_ids) == 0:
        continue  # skip if no label

    # Get CLS + merged word embeddings (last layer by default)
    cls_embedding, word_order, word_embs = get_cls_and_word_embs(
        text, tokenizer, model, device, l2norm=False, layer_idx=-1
    )

    # LIME target: first label (or you can use model's top predicted index instead)
    target = label_ids[0]
    try:
        num_features = min(max(len(word_order), 5), 20)  # stable features count
        exp = explainer.explain_instance(text, predict_proba, labels=[target], num_features=num_features)
        lime_list = exp.as_list(label=target)  # list of (token_str, weight)
        lime_tokens = [t for (t, w) in lime_list]
        lime_scores = [float(w) for (t, w) in lime_list]
    except Exception:
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

    # Save
    results.append({
        "text": text,
        "label_ids": label_ids,
        "cls_embedding": cls_embedding.cpu(),   # [H]
        "word_order": word_order,               # whitespace words
        "word_embeddings": word_embs.cpu(),     # [W,H] merged
        "lime_tokens": lime_tokens,
        "lime_scores": torch.tensor(lime_scores),
        "lime_scores_aligned": lime_scores_aligned,  # [W] aligned to word_order
    })

# Save
torch.save(results, LIME_OUTPUT_PATH)
print(f"✅ Saved merged word embeddings + LIME to {LIME_OUTPUT_PATH}")

# Quick peek
if results:
    ex = results[0]
    print("Example text:", ex["text"])
    print("Words:", ex["word_order"][:12])
    print("Word embeddings shape:", tuple(ex["word_embeddings"].shape))
    print("LIME aligned (first 12):", ex["lime_scores_aligned"][:12])

import torch, random

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
DATA_PATH = LIME_OUTPUT_PATH  # produced earlier
SAVE_CKPT = "plex_seismic_modernbert_lime.pth"

INPUT_SIZE = 768           # ModernBERT-base hidden size
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
            tau = 0.05
            keep = torch.abs(y) >= tau
            if keep.any():
                word_embs = word_embs[keep]
                y = y[keep]
            else:
                # if all tiny, keep the largest 1 token to avoid empty sentence
                j = torch.argmax(torch.abs(y))
                word_embs = word_embs[j:j+1]
                y = y[j:j+1]

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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

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
