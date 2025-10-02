# tools/generate_figures.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import numpy as np

os.makedirs("figures", exist_ok=True)
sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.1)

# 1) Model comparison (bar)
df_models = pd.DataFrame([
  {"model":"llama3.1:8b","F1":0.71,"NDCG@10":0.64,"MRR":0.59,"BLEU":0.28},
  {"model":"mistral:7b","F1":0.69,"NDCG@10":0.61,"MRR":0.56,"BLEU":0.26},
  {"model":"qwen2:7b","F1":0.73,"NDCG@10":0.66,"MRR":0.61,"BLEU":0.29},
  {"model":"gpt-4o-mini","F1":0.78,"NDCG@10":0.72,"MRR":0.67,"BLEU":0.33},
])
m = df_models.melt(id_vars="model", var_name="metric", value_name="score")
plt.figure(figsize=(7,4))
ax = sns.barplot(data=m, x="metric", y="score", hue="model")
ax.set_ylim(0,1); ax.set_ylabel("Score"); ax.set_xlabel("")
plt.legend(title="Model", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout(); plt.savefig("figures/model_comparison.svg"); plt.close()

# 2) Retrieval quality curve (NDCG@k)
k = np.array([1,3,5,10,20])
ndcg_llama = [0.48,0.58,0.62,0.66,0.68]
ndcg_mistral = [0.45,0.54,0.59,0.63,0.65]
plt.figure(figsize=(6,4))
plt.plot(k, ndcg_llama, marker='o', label='llama3.1:8b', lw=2, ms=6)
plt.plot(k, ndcg_mistral, marker='s', label='mistral:7b', lw=2, ms=6)
plt.xlabel('k'); plt.ylabel('NDCG@k'); plt.ylim(0,1); plt.grid(True, alpha=.3)
plt.legend(); plt.tight_layout(); plt.savefig("figures/ndcg_vs_k.svg"); plt.close()

# 3) PR and ROC curves (dummy example)
rng = np.random.default_rng(0)
y_true = rng.integers(0,2, size=500)
y_score = rng.random(500)*0.6 + (0.3*y_true)  # make positives score higher
prec, rec, _ = precision_recall_curve(y_true, y_score)
fpr, tpr, _ = roc_curve(y_true, y_score)
plt.figure(figsize=(6,4))
plt.plot(rec, prec, lw=2); plt.xlabel("Recall"); plt.ylabel("Precision")
plt.tight_layout(); plt.savefig("figures/pr_curve.svg"); plt.close()
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, lw=2, label=f"AUC={auc(fpr,tpr):.2f}")
plt.plot([0,1],[0,1],'--', color='gray'); plt.xlabel("FPR"); plt.ylabel("TPR")
plt.legend(); plt.tight_layout(); plt.savefig("figures/roc_curve.svg"); plt.close()

# 4) Ablation study (bar)
df_ablate = pd.DataFrame([
  {"setting":"Baseline","F1":0.63},
  {"setting":"+ NER signals","F1":0.67},
  {"setting":"+ Reranker","F1":0.71},
  {"setting":"+ CoT prompting","F1":0.74},
])
plt.figure(figsize=(6,4))
ax = sns.barplot(data=df_ablate, x="setting", y="F1", color="#4C72B0")
ax.set_ylim(0,1)
ax.bar_label(ax.containers[0], fmt="%.2f")
plt.xticks(rotation=20); plt.tight_layout(); plt.savefig("figures/ablation.svg"); plt.close()

# 5) Latency vs cost (bubble)
df_eff = pd.DataFrame([
  {"model":"llama3.1:8b","lat_ms":850,"cost_usd":0.0005,"F1":0.71},
  {"model":"mistral:7b","lat_ms":780,"cost_usd":0.0004,"F1":0.69},
  {"model":"qwen2:7b","lat_ms":720,"cost_usd":0.0004,"F1":0.73},
  {"model":"gpt-4o-mini","lat_ms":1100,"cost_usd":0.0012,"F1":0.78},
])
plt.figure(figsize=(6,4))
sizes = (df_eff["F1"]*10_0)**2
plt.scatter(df_eff["lat_ms"], df_eff["cost_usd"], s=sizes, alpha=.6)
for _, r in df_eff.iterrows():
    plt.text(r["lat_ms"]+10, r["cost_usd"], r["model"], fontsize=9)
plt.xlabel("Latency (ms)"); plt.ylabel("Cost per prompt (USD)")
plt.tight_layout(); plt.savefig("figures/latency_cost.svg"); plt.close()

# 6) Token usage (stacked)
df_tok = pd.DataFrame([
  {"model":"llama3.1:8b","in_tok":320,"out_tok":180},
  {"model":"mistral:7b","in_tok":310,"out_tok":170},
  {"model":"qwen2:7b","in_tok":305,"out_tok":190},
  {"model":"gpt-4o-mini","in_tok":340,"out_tok":210},
])
plt.figure(figsize=(6,4))
plt.bar(df_tok["model"], df_tok["in_tok"], label="Input tokens")
plt.bar(df_tok["model"], df_tok["out_tok"], bottom=df_tok["in_tok"], label="Output tokens")
plt.ylabel("Avg tokens"); plt.xticks(rotation=15); plt.legend()
plt.tight_layout(); plt.savefig("figures/tokens.svg"); plt.close()

print("Saved figures to ./figures/")