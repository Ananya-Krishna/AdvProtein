#!/usr/bin/env python3
"""
Generate basic statistics and exploratory plots for the protein dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

BASE_DIR = Path("/home/ark89/scratch_pi_ds256/ark89/Biomolecular-Optimization/AdvProtein")
DATA_CSV = BASE_DIR / "data/output/protein_dataset_final.csv"
FIGURE_DIR = BASE_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)

print("📊 Generating dataset report and figures...")

df = pd.read_csv(DATA_CSV)
folded = df[df["plddt_mean"].notna()].copy()

print(f"Total proteins: {len(df)}")
print(f"With structures: {len(folded)} ({len(folded)/len(df)*100:.1f}%)")

# === 1. pLDDT Distribution ===
plt.figure(figsize=(10, 6))
sns.histplot(data=folded, x="plddt_mean", bins=50, kde=True)
plt.title("Distribution of pLDDT (Structure Confidence)")
plt.xlabel("Mean pLDDT")
plt.ylabel("Count")
plt.axvline(x=50, color='red', linestyle='--', alpha=0.7, label="Typical threshold")
plt.legend()
plt.savefig(FIGURE_DIR / "plddt_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# === 2. Length vs pLDDT ===
plt.figure(figsize=(10, 6))
sns.scatterplot(data=folded, x="length", y="plddt_mean", alpha=0.6, s=15)
plt.title("Sequence Length vs Structure Confidence")
plt.xlabel("Sequence Length (aa)")
plt.ylabel("Mean pLDDT")
plt.savefig(FIGURE_DIR / "length_vs_plddt.png", dpi=300, bbox_inches='tight')
plt.close()

# === 3. Source breakdown ===
plt.figure(figsize=(8, 6))
source_counts = df["source"].value_counts()
source_counts.plot(kind='bar')
plt.title("Proteins by Source")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "source_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

# === 4. ESM2 fitness distribution ===
plt.figure(figsize=(10, 6))
sns.histplot(data=folded, x="esm2_logprob_mean", bins=50, kde=True)
plt.title("ESM-2 Log Probability Distribution (Fitness)")
plt.xlabel("ESM-2 Mean Log Probability")
plt.savefig(FIGURE_DIR / "esm2_fitness.png", dpi=300, bbox_inches='tight')
plt.close()

# Save summary statistics
summary = {
    "total_proteins": len(df),
    "folded": len(folded),
    "pct_folded": len(folded)/len(df)*100,
    "mean_plddt": folded["plddt_mean"].mean(),
    "median_plddt": folded["plddt_mean"].median(),
    "mean_length": df["length"].mean(),
    "sources": df["source"].value_counts().to_dict(),
    "high_quality_plddt": len(folded[folded["plddt_mean"] > 70]),
    "low_quality_plddt": len(folded[folded["plddt_mean"] < 40])
}

import json
with open(FIGURE_DIR / "dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ Figures and summary saved to {FIGURE_DIR}/")
print(f"   - plddt_distribution.png")
print(f"   - length_vs_plddt.png")
print(f"   - source_distribution.png")
print(f"   - esm2_fitness.png")
print(f"   - dataset_summary.json")

print("\nKey observations:")
print(f"- Average pLDDT is low ({folded['plddt_mean'].mean():.1f})")
print(f"- {len(folded[folded['plddt_mean'] < 40])} proteins have pLDDT < 40 (potentially noisy)")
print(f"- Sequence lengths vary widely (mean = {df['length'].mean():.1f} aa)")
