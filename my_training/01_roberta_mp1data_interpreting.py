# -*- coding: utf-8 -*-
"""
Exported from: PLEX_Training_Using_Modernbert_base_go_emotions.ipynb
Section: 🧪 Interpreting Metrics
Export time: 2025-11-08T20:37:11
Notes:
  - IPython magics (%) are commented out.
  - Shell commands (!) were redirected to install.sh or commented.
"""
# stress_test_prob_drop.py
import re, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

# --------------------------
# Config
# --------------------------
DATA_PATH = "mp1data_with_lime_words.pt"
PLEX_CKPT = "plex_seismic_modernbert_lime.pth"
MODEL_NAME = "s-nlp/roberta_toxicity_classifier"   # multi-label; sigmoid over 28 classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_LIST = [1, 3, 5]
SEED = 42
torch.manual_seed(SEED)

# --------------------------
# Load data (PyTorch 2.6 note)
# --------------------------
data = torch.load(DATA_PATH, weights_only=False)
print(f"Loaded {len(data)} samples.")

# --------------------------
# ModernBERT classifier
# --------------------------
tok = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
clf = RobertaForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)
clf.eval()
batch = tok.encode("You are amazing!", return_tensors="pt").to(DEVICE)

@torch.no_grad()
def pred_prob(text: str, target_idx: int) -> float:
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)
    logits = clf(**enc).logits
    probs = torch.sigmoid(logits)[0]  # [28]
    return float(probs[target_idx].item())

@torch.no_grad()
def top_pred_idx(text: str) -> int:
    enc = tok(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)
    logits = clf(**enc).logits
    probs = torch.sigmoid(logits)[0]  # [28]
    return int(torch.argmax(probs).item())

# --------------------------
# PLEX head (SeismicNet shared)
# --------------------------
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
    def __init__(self, input_size=768):
        super().__init__()
        self.net = SeismicNet(input_size)
    def forward(self, cls_emb: torch.Tensor, word_embs: torch.Tensor) -> torch.Tensor:
        """
        cls_emb: [W,H] or [H] (we'll expand if needed)
        word_embs: [W,H]
        returns: [W] cosine sims
        """
        single = False
        if cls_emb.dim() == 1:
            # repeat cls per word
            cls_emb = cls_emb.unsqueeze(0).repeat(word_embs.shape[0], 1)
            single = True
        cls_proj  = self.net(cls_emb)        # [W,64]
        word_proj = self.net(word_embs)      # [W,64]
        # Center token projections (helps dispersion)
        word_proj = word_proj - word_proj.mean(dim=0, keepdim=True)
        cls_norm  = F.normalize(cls_proj, dim=-1)
        word_norm = F.normalize(word_proj, dim=-1)
        sims = (cls_norm * word_norm).sum(dim=-1)  # [W]
        return sims

# load PLEX weights
plex = PLEXHeadShared(input_size=768).to(DEVICE)
state = torch.load(PLEX_CKPT, map_location="cpu")
sd = state.get("state_dict", state)
# If saved as bare SeismicNet keys, wrap:
if isinstance(sd, dict) and all(k.split(".")[0] in {"net","fc1","fc2"} for k in sd.keys()):
    sd = { (f"net.{k}" if k.startswith(("fc1","fc2")) else k): v for k,v in sd.items() }
missing, unexpected = plex.load_state_dict(sd, strict=False)
print("PLEX load -> missing:", missing)
print("PLEX load -> unexpected:", unexpected)
plex.eval()

@torch.no_grad()
def plex_scores_for_entry(entry) -> np.ndarray:
    we: torch.Tensor = entry["word_embeddings"]   # [W,H]
    if we.numel() == 0:
        return np.array([])
    cls: torch.Tensor = entry["cls_embedding"]    # [H]
    scores = plex(cls.to(DEVICE), we.to(DEVICE)).detach().cpu().numpy()
    return scores

# --------------------------
# Utils: remove top-k words (whitespace tokens)
# --------------------------
def remove_words(text: str, words: list[str], top_indices: np.ndarray, k: int) -> str:
    ws = words[:]  # already whitespace-split used in extraction
    pick = set(top_indices[:min(k, len(top_indices))].tolist())
    kept = [w for i, w in enumerate(ws) if i not in pick]
    out = " ".join(kept)
    out = re.sub(r"\s+", " ", out).strip()
    # avoid empty string: fall back to original if we removed everything
    return out if out else text

def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0:
        return np.array([], dtype=int)
    return np.argsort(-scores)[:min(k, scores.size)]

# --------------------------
# Evaluate probability drop
# --------------------------
def mean_ci(x):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, (np.nan, np.nan)
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(len(x))) if len(x) > 1 else np.nan
    return m, (m - 1.96*se, m + 1.96*se) if not np.isnan(se) else (np.nan, np.nan)

results = {k: {"lime": [], "plex": []} for k in K_LIST}
n_used = 0

for ex in data:
    words = ex.get("word_order", [])
    if not words:
        continue
    lime = ex.get("lime_scores_aligned", torch.tensor([]))
    if lime.numel() == 0:
        continue
    lime_np = lime.numpy()

    # PLEX scores for this entry
    plex_np = plex_scores_for_entry(ex)
    if plex_np.size == 0:
        continue

    text = ex["text"]

    # use model's own top predicted class on ORIGINAL sentence
    try:
        tgt_idx = top_pred_idx(text)
        p0 = pred_prob(text, tgt_idx)
    except Exception:
        continue

    for k in K_LIST:
        # LIME top-k removal
        li = topk_indices(lime_np, k)
        text_l = remove_words(text, words, li, k)
        p_l = pred_prob(text_l, tgt_idx)
        results[k]["lime"].append(p0 - p_l)

        # PLEX top-k removal
        pi = topk_indices(plex_np, k)
        text_p = remove_words(text, words, pi, k)
        p_p = pred_prob(text_p, tgt_idx)
        results[k]["plex"].append(p0 - p_p)

    n_used += 1

print(f"\nEvaluated {n_used} sentences.")

# Report means + 95% CI
for k in K_LIST:
    m_l, ci_l = mean_ci(results[k]["lime"])
    m_p, ci_p = mean_ci(results[k]["plex"])
    print(f"\nTop-{k} removal Δprob (mean [95% CI]):")
    print(f"  LIME: {m_l:.4f}  [{ci_l[0]:.4f}, {ci_l[1]:.4f}]")
    print(f"  PLEX: {m_p:.4f}  [{ci_p[0]:.4f}, {ci_p[1]:.4f}]")
