import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_metadata, clean_metadata, common_title_words, ensure_dir

sns.set(style="whitegrid")

def analyze(input_path: str, out_dir: str = "reports", sample: int = 0):
    out = ensure_dir(out_dir)

    # Load + optional sample
    df = load_metadata(input_path)
    if sample and len(df) > sample:
        df = df.sample(n=sample, random_state=42)

    # Clean
    df_clean = clean_metadata(df)
    df_clean.to_csv(out / "cleaned_metadata.csv", index=False)

    # Analysis
    by_year = df_clean["year"].value_counts().sort_index()
    top_journals = df_clean["journal"].value_counts().head(10) if "journal" in df_clean.columns else pd.Series(dtype=int)
    by_source = df_clean["source_x"].value_counts().head(10) if "source_x" in df_clean.columns else pd.Series(dtype=int)
    title_words = common_title_words(df_clean, top_n=20)

    # Charts
    if not by_year.empty:
        fig1, ax1 = plt.subplots()
        ax1.plot(by_year.index, by_year.values, marker="o")
        ax1.set_title("CORD-19 Publications by Year")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Paper Count")
        fig1.tight_layout()
        fig1.savefig(out/"publications_by_year.png", dpi=150)

    if not top_journals.empty:
        fig2, ax2 = plt.subplots()
        top_journals.sort_values(ascending=True).plot(kind="barh", ax=ax2)
        ax2.set_title("Top Publishing Journals (Top 10)")
        fig2.tight_layout()
        fig2.savefig(out/"top_journals.png", dpi=150)

    if "title_word_count" in df_clean.columns:
        fig3, ax3 = plt.subplots()
        df_clean["title_word_count"].plot(kind="hist", bins=30, ax=ax3)
        ax3.set_title("Title Word Count Distribution")
        fig3.tight_layout()
        fig3.savefig(out/"title_word_count_hist.png", dpi=150)

    if not by_source.empty:
        fig4, ax4 = plt.subplots()
        by_source.sort_values(ascending=True).plot(kind="barh", ax=ax4)
        ax4.set_title("Top Sources (Top 10)")
        fig4.tight_layout()
        fig4.savefig(out/"top_sources.png", dpi=150)

    # Insights
    with open(out/"summary.txt", "w", encoding="utf-8") as f:
        f.write("Key Insights\n")
        f.write("============\n\n")
        if not by_year.empty:
            f.write(f"- Peak year: {int(by_year.idxmax())} with {int(by_year.max())} papers\n")
        if not top_journals.empty:
            f.write(f"- Most prolific journal: {top_journals.index[0]}\n")
        if not title_words.empty:
            f.write(f"- Common title words: {', '.join(title_words['word'].head(10).tolist())}\n")

    print(f"Analysis complete. Reports saved to: {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", "-i", default="data/metadata.csv", help="Path to metadata.csv")
    p.add_argument("--out", "-o", default="reports", help="Output directory")
    p.add_argument("--sample", type=int, default=0, help="Optional sample size (e.g., 50000)")
    args = p.parse_args()
    analyze(args.input, args.out, args.sample)
