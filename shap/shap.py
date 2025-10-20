import torch
import shap
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import glob
import os

# ✅ 1️⃣ 加载模型（binary toxic classifier）
model_name = "SkolkovoInstitute/roberta_toxicity_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# ✅ 2️⃣ 定义预测函数
def predict_toxic(texts):
    if isinstance(texts, str):
        texts = [texts]
    elif isinstance(texts, (tuple, np.ndarray)):
        texts = list(texts)
    elif not isinstance(texts, list):
        try:
            texts = list(texts)
        except TypeError:
            raise ValueError("Unsupported input type for predict_toxic")

    texts = [str(text) for text in texts]

    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
    return probs[:, 1].unsqueeze(1).cpu().numpy()

# ✅ 3️⃣ 定义 SHAP masker & explainer
masker = shap.maskers.Text(tokenizer)
explainer = shap.Explainer(predict_toxic, masker)

# ✅ 4️⃣ 获取所有 subreddit CSV 文件路径
csv_files = [
    "2007scape.csv",
    "CFB.csv",
    "DIY.csv",
    "HistoryPorn.csv",
    "Showerthoughts.csv",
    "australia.csv",
    "aww.csv",
    "canada.csv",
    "depression.csv",
    "funny.csv"
]  # 例如 subreddit1.csv, subreddit2.csv, ...

# ✅ 5️⃣ 循环处理每个文件
for file_path in csv_files:
    subreddit_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n🚀 Processing {subreddit_name} ...")

    df = pd.read_csv(file_path)
    df["body"] = df["body"].fillna("")
    texts_to_explain = df["body"].tolist()[:100]  # 可调整数量

    print(f"🧩 Explaining {len(texts_to_explain)} comments from {subreddit_name}...")

    shap_values = explainer(texts_to_explain)

    rows = []
    for i, text in enumerate(texts_to_explain):
        toks = np.array(shap_values.data[i])
        vals = np.array(shap_values.values[i]).flatten()
        for t, v in zip(toks, vals):
            if t.strip() not in ["[CLS]", "[SEP]", "[PAD]", ""]:
                rows.append({
                    "Subreddit": subreddit_name,
                    "Sentence_ID": i + 1,
                    "Text": text,
                    "Token": t,
                    "SHAP_Impact_on_Toxic": float(v),
                    "Interpretation": "↑ increase toxic" if v > 0 else "↓ decrease toxic"
                })

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(["Sentence_ID", "SHAP_Impact_on_Toxic"], ascending=[True, False])

    out_name = f"toxic_shap_table_{subreddit_name}.csv"
    df_out.to_csv(out_name, index=False, encoding="utf-8-sig")
    print(f"💾 Saved as {out_name}")

print("\n✅ All subreddit files processed successfully!")