import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import load_metadata, clean_metadata, common_title_words

st.set_page_config(page_title="CORD-19 Data Explorer", layout="wide")
sns.set(style="whitegrid")

st.title("CORD-19 Data Explorer")
st.caption("Simple exploration of COVID-19 research papers")

# File input
default_path = "data/metadata.csv"
path = st.text_input("Path to metadata CSV", value=default_path)

# Load data with caching
@st.cache_data
def load_and_clean(p):
    df = load_metadata(p)
    df = clean_metadata(df)
    return df

try:
    df = load_and_clean(path)
except Exception as e:
    st.error(f"Could not load file: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
if "year" in df.columns and df["year"].notna().any():
    years = sorted([int(y) for y in df["year"].dropna().unique()])
    yr = st.sidebar.slider("Year range", min_value=min(years), max_value=max(years), value=(min(years), max(years)))
    df_f = df[(df["year"] >= yr[0]) & (df["year"] <= yr[1])].copy()
else:
    df_f = df.copy()

if "journal" in df_f.columns:
    journals = st.sidebar.multiselect("Journals (top 50 shown)", sorted(df_f["journal"].dropna().unique().tolist())[:50])
    if journals:
        df_f = df_f[df_f["journal"].isin(journals)]

# Top area: metrics + sample
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df_f):,}")
c2.metric("Columns", f"{df_f.shape[1]}")
missing = df_f.isnull().sum().sum()
c3.metric("Missing values", f"{int(missing):,}")

st.subheader("Sample of data")
st.dataframe(df_f.head(50))

# Charts
st.subheader("Publications over time")
if "year" in df_f.columns and df_f["year"].notna().any():
    by_year = df_f["year"].value_counts().sort_index()
    fig1, ax1 = plt.subplots()
    ax1.plot(by_year.index, by_year.values, marker="o")
    ax1.set_title("Publications by Year")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)
else:
    st.info("No year data available")

st.subheader("Top journals")
if "journal" in df_f.columns:
    top_j = df_f["journal"].value_counts().head(10)
    fig2, ax2 = plt.subplots()
    top_j.sort_values(ascending=True).plot(kind="barh", ax=ax2)
    ax2.set_title("Top Publishing Journals")
    ax2.set_xlabel("Paper Count")
    ax2.set_ylabel("Journal")
    st.pyplot(fig2)
else:
    st.info("No journal column found")

st.subheader("Title word count distribution")
if "title_word_count" in df_f.columns:
    fig3, ax3 = plt.subplots()
    df_f["title_word_count"].plot(kind="hist", bins=30, ax=ax3)
    ax3.set_title("Title Word Count")
    ax3.set_xlabel("Words")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3)

# Common words
st.subheader("Common words in titles")
top_words = common_title_words(df_f, top_n=20)
st.dataframe(top_words)

# Download filtered data
st.download_button(
    "Download filtered data as CSV",
    data=df_f.to_csv(index=False).encode("utf-8"),
    file_name="filtered_metadata.csv",
    mime="text/csv",
)
