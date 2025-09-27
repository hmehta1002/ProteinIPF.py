import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import py3Dmol
import requests
import os

# ------------------------
# Paths for sample files
# ------------------------
SAMPLE_CSV_PATH = "samples/ipf_sample.csv"
SAMPLE_FASTA_PATH = "samples/ipf_sequences.fasta"

# ------------------------
# App Title
# ------------------------
st.set_page_config(page_title="Protein Network Explorer", layout="wide")

st.markdown(
    "<h1 style='text-align:center; color:#6A0DAD;'>🧬 Protein Network Explorer — Full</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Explore human protein networks, sequences, motifs, stability & 3D structures.</p>",
    unsafe_allow_html=True
)

# ------------------------
# Sidebar Upload
# ------------------------
st.sidebar.header("📂 Upload Protein Network CSV")

uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv", key="file_uploader")

# Allow reset to sample
if st.sidebar.button("Use Sample CSV", key="reset_button"):
    uploaded_file = SAMPLE_CSV_PATH if os.path.exists(SAMPLE_CSV_PATH) else None

# ------------------------
# Load Data
# ------------------------
df = None
if uploaded_file is not None:
    try:
        if isinstance(uploaded_file, str):  # sample path
            df = pd.read_csv(uploaded_file)
        else:  # user upload
            df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"✅ File loaded: {uploaded_file if isinstance(uploaded_file,str) else uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"❌ Could not read file: {e}")

# ------------------------
# Tabs
# ------------------------
tabs = st.tabs([
    "Upload / Files",
    "Network Map (Clusters)",
    "Network Metrics",
    "Protein Details",
    "Sequences",
    "Motifs / Domains",
    "3D Structure",
    "Intrinsic Disorder & Stability"
])

# ------------------------
# Upload Tab
# ------------------------
with tabs[0]:
    st.subheader("📂 File Upload & Samples")

    if df is not None:
        st.write("### Preview of Uploaded CSV")
        st.dataframe(df.head())
        st.sidebar.markdown("**Detected CSV Columns:**")
        try:
            st.sidebar.write(list(df.columns))
        except Exception as e:
            st.sidebar.write(f"Could not read columns: {e}")
    else:
        st.info("No file loaded yet. Please upload a CSV or use the sample.")

    # Download sample files
    if os.path.exists(SAMPLE_CSV_PATH):
        with open(SAMPLE_CSV_PATH, "rb") as f:
            st.download_button("⬇️ Download Sample CSV", f, file_name="sample.csv", key="dl_csv")
    if os.path.exists(SAMPLE_FASTA_PATH):
        with open(SAMPLE_FASTA_PATH, "rb") as f:
            st.download_button("⬇️ Download Sample FASTA", f, file_name="sample.fasta", key="dl_fasta")

# ------------------------
# Other Tabs (skeletons preserved, safe from errors)
# ------------------------
with tabs[1]:
    st.subheader("🌐 Network Map with Clusters")
    if df is not None:
        G = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])
        net = Network(notebook=False)
        net.from_nx(G)
        net.show("network.html")
        st.components.v1.html(open("network.html").read(), height=500)
    else:
        st.warning("Please upload a CSV first.")

with tabs[2]:
    st.subheader("📊 Network Metrics")
    if df is not None:
        G = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1])
        metrics = {
            "Closeness Centrality": nx.closeness_centrality(G),
            "Betweenness Centrality": nx.betweenness_centrality(G),
            "Clustering Coefficient": nx.clustering(G),
        }
        st.json(metrics)
    else:
        st.warning("Please upload a CSV first.")

with tabs[3]:
    st.subheader("🔎 Protein Details")
    st.info("Protein metadata integration goes here.")

with tabs[4]:
    st.subheader("🧾 Sequences")
    st.info("FASTA-based sequence fetch and display here.")

with tabs[5]:
    st.subheader("🧩 Motifs / Domains")
    st.info("Motif/domain integration here.")

with tabs[6]:
    st.subheader("🔬 3D Structure (AlphaFold)")
    st.info("3Dmol viewer integration here.")

with tabs[7]:
    st.subheader("📉 Intrinsic Disorder & Stability")
    st.info("Disorder and stability predictions here.")
