# -*- coding: utf-8 -*-
"""
PLEX Model Demo - 展示训练的 PLEX 模型输出
演示如何使用训练好的 PLEX 模型来解释文本
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Tuple
import matplotlib.pyplot as plt
import os
import re
import string
from datetime import datetime
from lime.lime_text import LimeTextExplainer

# --------------------------
# Config
# --------------------------
DATA_PATH = "askhistorians_softmax_data_with_lime_words.pt"  # 可选：从保存的数据文件加载
PLEX_CKPT = "askhistorians_plex_seismic_modernbert_lime.pth"
# 指定你要加载的目录
# 如果想加载手动 early-stop 保存的最好模型：
MODEL_DIR = "askhistorians"
# 如果想用最后一次 trainer.save_model 的版本：
# MODEL_DIR = "finetuned-roberta-toxicity/games"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 5  # 显示 top-k 重要的词
SEED = 42
torch.manual_seed(SEED)
OUTPUT_DIR = "plex_visualizations"  # 保存图片的目录

# --------------------------
# PLEX Model
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
        if cls_emb.dim() == 1:
            cls_emb = cls_emb.unsqueeze(0).repeat(word_embs.shape[0], 1)
        cls_proj = self.net(cls_emb)        # [W,64]
        word_proj = self.net(word_embs)     # [W,64]
        # Center token projections (helps dispersion)
        word_proj = word_proj - word_proj.mean(dim=0, keepdim=True)
        cls_norm = F.normalize(cls_proj, dim=-1)
        word_norm = F.normalize(word_proj, dim=-1)
        sims = (cls_norm * word_norm).sum(dim=-1)  # [W]
        return sims

# --------------------------
# Load PLEX Model
# --------------------------
print("Loading PLEX model...")
plex = PLEXHeadShared(input_size=768).to(DEVICE)
state = torch.load(PLEX_CKPT, map_location="cpu")
sd = state.get("state_dict", state)
# If saved as bare SeismicNet keys, wrap:
if isinstance(sd, dict) and all(k.split(".")[0] in {"net", "fc1", "fc2"} for k in sd.keys()):
    sd = {(f"net.{k}" if k.startswith(("fc1", "fc2")) else k): v for k, v in sd.items()}
missing, unexpected = plex.load_state_dict(sd, strict=False)
print(f"PLEX load -> missing: {missing}")
print(f"PLEX load -> unexpected: {unexpected}")
plex.eval()
print("✅ PLEX model loaded successfully!")

# --------------------------
# Load Classifier Model
# --------------------------
print(f"\nLoading classifier model from: {MODEL_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
classifier = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
classifier.eval()
classifier.to(DEVICE)
num_labels = classifier.config.num_labels
print("✅ Classifier model loaded successfully!")

# --------------------------
# LIME Setup
# --------------------------
print("\nSetting up LIME explainer...")

# LIME prediction function
# Note: Model was trained with CrossEntropyLoss and num_labels=2, so use softmax
def predict_proba(texts):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256).to(DEVICE)
    with torch.no_grad():
        logits = classifier(**tokens).logits
        # Binary classification with 2 outputs: use softmax (not sigmoid)
        # probs[0] = P(class 0), probs[1] = P(class 1)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return probs

# LIME explainer
lime_explainer = LimeTextExplainer(class_names=[str(i) for i in range(num_labels)])
print("✅ LIME explainer ready!")

# --------------------------
# Helper: Normalize token for alignment
# --------------------------
def normalize_token(s: str) -> str:
    """Normalize token for matching"""
    s = s.lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", s).strip()

# --------------------------
# Helper Functions
# --------------------------
def get_cls_and_word_embs(text: str, tokenizer, model, device, l2norm=False, layer_idx=-1):
    """
    Extract CLS embedding and word embeddings from text.
    Returns:
      cls_embedding: [H] tensor
      word_order: list[str] - whitespace-split words
      word_embs: [W,H] tensor aligned to word_order
    """
    # Pass 1: get actual tokenized length
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
    offsets = enc["offset_mapping"][0].tolist()

    with torch.no_grad():
        out = model(
            **{k: v for k, v in enc.items() if k in ["input_ids", "attention_mask"]},
            output_hidden_states=True
        )
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
            w_start = char_ptr
        w_end = w_start + len(w)
        words_char_spans.append((w_start, w_end, w))
        char_ptr = w_end

    for (w_start, w_end, w) in words_char_spans:
        # subwords overlapping the word's char span
        embs = [emb for (s, e), emb in sub_spans if not (e <= w_start or s >= w_end)]
        if len(embs) == 0:
            # relaxed fallback
            embs = [emb for (s, e), emb in sub_spans if (text[s:e] in w) or (w in text[s:e])]
        if len(embs) > 0:
            embs = torch.stack(embs, dim=0)
            word_emb = embs.mean(dim=0)
            word_emb_list.append(word_emb)

    word_embs = torch.stack(word_emb_list, dim=0) if len(word_emb_list) > 0 else torch.empty(0, hidden.shape[-1])
    return cls_embedding, word_order, word_embs

@torch.no_grad()
def get_plex_scores(text: str) -> Tuple[List[str], np.ndarray]:
    """
    Get PLEX scores for a text.
    Returns:
      word_order: list of words
      scores: numpy array of PLEX scores for each word
    """
    cls_emb, word_order, word_embs = get_cls_and_word_embs(
        text, tokenizer, classifier, DEVICE, l2norm=False, layer_idx=-1
    )
    
    if word_embs.numel() == 0:
        return word_order, np.array([])
    
    scores = plex(cls_emb.to(DEVICE), word_embs.to(DEVICE)).detach().cpu().numpy()
    return word_order, scores

@torch.no_grad()
def get_lime_scores(text: str, word_order: List[str], target_label: int = None) -> np.ndarray:
    """
    Get LIME scores for a text, aligned to word_order.
    Returns:
      lime_scores_aligned: numpy array of LIME scores aligned to word_order
    """
    if len(word_order) == 0:
        return np.array([])
    
    # Determine target label
    if target_label is None:
        # Use model's top predicted class (same as 00_roberta_mp1data_plex_guide.py behavior)
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            logits = classifier(**tokens).logits
            # Binary classification with 2 outputs: use softmax (not sigmoid)
            probs = torch.softmax(logits, dim=-1)[0]  # [2] probabilities
            target_label = int(torch.argmax(probs).item())
    
    try:
        # Same as 00_roberta_mp1data_plex_guide.py: stable features count
        num_features = min(max(len(word_order), 5), 20)
        exp = lime_explainer.explain_instance(text, predict_proba, labels=[target_label], num_features=num_features)
        lime_list = exp.as_list(label=target_label)  # list of (token_str, weight)
        lime_tokens = [t for (t, w) in lime_list]
        lime_scores = [float(w) for (t, w) in lime_list]
    except Exception as e:
        # Same as 00_roberta_mp1data_plex_guide.py: return empty/zeros on failure
        print(f"  ⚠️  LIME failed: {e}")
        return np.zeros(len(word_order))
    
    # Align LIME tokens to whitespace words (same as 00_roberta_mp1data_plex_guide.py)
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
        # else: no match → ignore (punctuation/mismatch), same as original
    
    return np.array(aligned)

def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """返回绝对值最大的 top-k 索引（处理正负分数）"""
    if scores.size == 0:
        return np.array([], dtype=int)
    return np.argsort(-np.abs(scores))[:min(k, scores.size)]

def plot_word_importances(words, scores, title, save_path=None, words_per_row=8, fontsize=13):
    """
    绘制词重要性可视化图
    Args:
        words: 词列表
        scores: 分数数组
        title: 图表标题
        save_path: 保存路径，如果为None则只显示不保存
        words_per_row: 每行显示的词数
        fontsize: 字体大小
    """
    if len(words) == 0:
        print(f"[{title}] No words to display.")
        return
    
    scores = np.asarray(scores, dtype=float)
    m = np.max(np.abs(scores)) if scores.size else 1.0
    s = scores / (m + 1e-8)
    
    cmap_neg = plt.get_cmap("Blues")
    cmap_pos = plt.get_cmap("Reds")
    
    plt.figure(figsize=(12, 2.4))
    x, y = 0.0, 0.8
    
    for i, (w, v) in enumerate(zip(words, s)):
        color = cmap_pos(v) if v >= 0 else cmap_neg(-v)
        plt.text(x, y, w, fontsize=fontsize, ha='left', va='center', color='black',
                 bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3'))
        x += (len(w) * 0.012) + 0.05
        if (i + 1) % words_per_row == 0:
            x = 0.0
            y -= 0.25
    
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  ✅ Saved visualization to: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_lime_and_plex_comparison(words, lime_scores, plex_scores, text, save_path=None, words_per_row=8, fontsize=13):
    """
    同时绘制 LIME 和 PLEX 的词重要性可视化对比图
    Args:
        words: 词列表
        lime_scores: LIME 分数数组
        plex_scores: PLEX 分数数组
        text: 原始文本
        save_path: 保存路径，如果为None则只显示不保存
        words_per_row: 每行显示的词数
        fontsize: 字体大小
    """
    if len(words) == 0:
        print(f"No words to display.")
        return
    
    lime_scores = np.asarray(lime_scores, dtype=float)
    plex_scores = np.asarray(plex_scores, dtype=float)
    
    # Normalize scores separately
    m_lime = np.max(np.abs(lime_scores)) if lime_scores.size and np.max(np.abs(lime_scores)) > 0 else 1.0
    m_plex = np.max(np.abs(plex_scores)) if plex_scores.size and np.max(np.abs(plex_scores)) > 0 else 1.0
    
    s_lime = lime_scores / (m_lime + 1e-8)
    s_plex = plex_scores / (m_plex + 1e-8)
    
    cmap_neg = plt.get_cmap("Blues")
    cmap_pos = plt.get_cmap("Reds")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4.8))
    
    # Plot LIME
    x, y = 0.0, 0.8
    for i, (w, v) in enumerate(zip(words, s_lime)):
        color = cmap_pos(v) if v >= 0 else cmap_neg(-v)
        ax1.text(x, y, w, fontsize=fontsize, ha='left', va='center', color='black',
                 bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3'))
        x += (len(w) * 0.012) + 0.05
        if (i + 1) % words_per_row == 0:
            x = 0.0
            y -= 0.25
    
    ax1.set_title(f"LIME Word Importances: {text[:60]}..." if len(text) > 60 else f"LIME Word Importances: {text}")
    ax1.axis('off')
    
    # Plot PLEX
    x, y = 0.0, 0.8
    for i, (w, v) in enumerate(zip(words, s_plex)):
        color = cmap_pos(v) if v >= 0 else cmap_neg(-v)
        ax2.text(x, y, w, fontsize=fontsize, ha='left', va='center', color='black',
                 bbox=dict(facecolor=color, edgecolor='none', boxstyle='round,pad=0.3'))
        x += (len(w) * 0.012) + 0.05
        if (i + 1) % words_per_row == 0:
            x = 0.0
            y -= 0.25
    
    ax2.set_title(f"PLEX Word Importances: {text[:60]}..." if len(text) > 60 else f"PLEX Word Importances: {text}")
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"  ✅ Saved comparison visualization to: {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()

def print_plex_output(text: str, word_order: List[str], plex_scores: np.ndarray, 
                     lime_scores: np.ndarray = None, top_k: int = TOP_K, 
                     save_plot: bool = True, plot_filename: str = None):
    """打印 PLEX 和 LIME 输出结果"""
    print("\n" + "="*80)
    print("Text:")
    print(f"  {text}")
    print("\n" + "-"*80)
    
    if plex_scores.size == 0:
        print("No PLEX scores available.")
        return
    
    # PLEX Top-k important words
    top_k_indices_plex = topk_indices(plex_scores, top_k)
    print(f"\nTop-{top_k} Most Important Words (PLEX):")
    print("-"*80)
    for i, idx in enumerate(top_k_indices_plex, 1):
        word = word_order[idx] if idx < len(word_order) else "N/A"
        score = plex_scores[idx]
        print(f"  {i:2d}. {word:20s} | PLEX Score: {score:7.4f} | Abs: {abs(score):7.4f}")
    
    # LIME Top-k if available
    if lime_scores is not None and lime_scores.size > 0:
        top_k_indices_lime = topk_indices(lime_scores, top_k)
        print(f"\nTop-{top_k} Most Important Words (LIME):")
        print("-"*80)
        for i, idx in enumerate(top_k_indices_lime, 1):
            word = word_order[idx] if idx < len(word_order) else "N/A"
            score = lime_scores[idx]
            print(f"  {i:2d}. {word:20s} | LIME Score: {score:7.4f} | Abs: {abs(score):7.4f}")
    
    # Statistics
    print(f"\nStatistics:")
    print("-"*80)
    print(f"  Total words: {len(word_order)}")
    print(f"  PLEX - Score range: [{plex_scores.min():.4f}, {plex_scores.max():.4f}]")
    print(f"  PLEX - Score mean: {plex_scores.mean():.4f}, std: {plex_scores.std():.4f}")
    if lime_scores is not None and lime_scores.size > 0:
        print(f"  LIME - Score range: [{lime_scores.min():.4f}, {lime_scores.max():.4f}]")
        print(f"  LIME - Score mean: {lime_scores.mean():.4f}, std: {lime_scores.std():.4f}")
    
    # Plot visualization
    if save_plot and plex_scores.size > 0:
        if plot_filename is None:
            # 生成默认文件名：使用时间戳和文本前30个字符
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            text_safe = "".join(c for c in text[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            text_safe = text_safe.replace(' ', '_')[:30]
            plot_filename = f"comparison_{timestamp}_{text_safe}.png"
        
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, plot_filename)
        
        # If LIME scores available, plot comparison; otherwise just PLEX
        if lime_scores is not None and lime_scores.size > 0:
            plot_lime_and_plex_comparison(word_order, lime_scores, plex_scores, text, 
                                         save_path=save_path)
        else:
            plot_word_importances(word_order, plex_scores, 
                                f"PLEX Word Importances: {text[:50]}...", 
                                save_path=save_path)

# --------------------------
# Demo: Load from saved data
# --------------------------
print("\n" + "="*80)
print("PLEX Model Demo")
print("="*80)

# Create output directory for visualizations
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\n📁 Output directory for visualizations: {OUTPUT_DIR}")

# # Option 1: Load from saved data file
# try:
#     print(f"\nLoading data from {DATA_PATH}...")
#     data = torch.load(DATA_PATH, weights_only=False)
#     print(f"✅ Loaded {len(data)} samples from saved data.")
    
#     # Process first few examples
#     num_examples = min(3, len(data))
#     print(f"\nProcessing first {num_examples} examples from saved data:")
#     print("-"*80)
    
#     for i in range(num_examples):
#         ex = data[i]
#         text = ex["text"]
#         word_order = ex.get("word_order", [])
        
#         # Get PLEX scores from saved embeddings
#         cls_emb = ex["cls_embedding"]
#         word_embs = ex["word_embeddings"]
        
#         if word_embs.numel() == 0:
#             print(f"\nExample {i+1}: Skipped (no word embeddings)")
#             continue
        
#         with torch.no_grad():
#             plex_scores = plex(cls_emb.to(DEVICE), word_embs.to(DEVICE)).detach().cpu().numpy()
        
#         # Get LIME scores (run LIME analysis)
#         print(f"\n  Running LIME analysis for example {i+1}...")
#         target_label = ex.get("label_ids", [None])[0] if ex.get("label_ids") else None
#         lime_scores = get_lime_scores(text, word_order, target_label=target_label)
        
#         plot_filename = f"saved_data_example_{i+1}.png"
#         print_plex_output(text, word_order, plex_scores, lime_scores=lime_scores, 
#                          top_k=TOP_K, save_plot=True, plot_filename=plot_filename)
        
# except FileNotFoundError:
#     print(f"⚠️  Data file {DATA_PATH} not found. Skipping saved data examples.")

# --------------------------
# Demo: Process custom text examples
# --------------------------
print("\n" + "="*80)
print("Processing Custom Text Examples")
print("="*80)

# Custom text examples to demonstrate PLEX
custom_texts = [
    "Well considering they threw their own shit and piss into the streets I'm guessing they did the same",
    # "That was terrible. I hated every minute of it.",
    # "The food was okay, nothing special.",
]

for i, text in enumerate(custom_texts, 1):
    print(f"\n\nExample {i}:")
    word_order, plex_scores = get_plex_scores(text)
    if plex_scores.size > 0:
        # Get LIME scores
        print(f"  Running LIME analysis...")
        lime_scores = get_lime_scores(text, word_order, target_label=None)
        
        plot_filename = f"custom_example_{i}.png"
        print_plex_output(text, word_order, plex_scores, lime_scores=lime_scores, 
                         top_k=TOP_K, save_plot=True, plot_filename=plot_filename)
    else:
        print(f"Failed to get PLEX scores for: {text}")

print("\n" + "="*80)
print("Demo completed!")
print(f"All visualizations saved to: {OUTPUT_DIR}/")
print("="*80)

