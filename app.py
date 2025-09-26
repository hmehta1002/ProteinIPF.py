import streamlit as st
import pandas as pd
import requests
import networkx as nx
from pyvis.network import Network
import py3Dmol
import matplotlib.pyplot as plt
from io import StringIO
from Bio import SeqIO

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Protein Network Explorer", layout="wide")

st.title("🧬 Protein Network Explorer")
st.markdown("Explore human protein networks, sequences, motifs, stability & 3D structures.")

# -----------------------------
# Sample files (embedded in repo)
# -----------------------------
SAMPLE_CSV_PATH = "samples/ipf_sample.csv"
SAMPLE_FASTA_PATH = "samples/ipf_sequences.fasta"

with open(SAMPLE_CSV_PATH, "rb") as f:
    st.sidebar.download_button("⬇️ Download Sample CSV", f, "ipf_sample.csv", key="sample_csv_btn")

with open(SAMPLE_FASTA_PATH, "rb") as f:
    st.sidebar.download_button("⬇️ Download Sample FASTA", f, "ipf_sequences.fasta", key="sample_fasta_btn")

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "📂 Upload / Files",
    "🗺️ Network Map (Clusters)",
    "📊 Network Metrics",
    "🧾 Protein Details",
    "🧬 Sequences",
    "📑 Motifs / Domains",
    "🔬 3D Structure",
    "🧩 Intrinsic Disorder & Stability"
])

# -----------------------------
# Tab 1: Upload / Files
# -----------------------------
with tabs[0]:
    st.header("📂 Upload your Protein Network Data")

    uploaded_file = st.file_uploader("Upload a CSV file (with Protein1, Protein2 columns)", type=["csv"], key="uploader")
    df = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded file with {len(df)} rows")
        st.dataframe(df.head(), use_container_width=True)

    else:
        # load sample file if none uploaded
        df = pd.read_csv(SAMPLE_CSV_PATH)
        st.info("No file uploaded. Using sample CSV.")
        st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# Tab 2: Network Map (Clusters)
# -----------------------------
with tabs[1]:
    st.header("🗺️ Protein Interaction Network with Clusters")

    if df is not None:
        # Build network
        G = nx.Graph()
        for _, row in df.iterrows():
            try:
                p1, p2 = row["Protein1"], row["Protein2"]
                G.add_edge(p1, p2)
            except KeyError:
                st.error("CSV must contain columns 'Protein1' and 'Protein2'")
                break

        # Community detection (clustering)
        communities = nx.algorithms.community.greedy_modularity_communities(G)
        comm_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                comm_map[node] = i

        # Pyvis interactive network
        net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
        for node in G.nodes():
            net.add_node(node, title=node, color=plt.cm.tab20(comm_map[node] % 20))

        for edge in G.edges():
            net.add_edge(edge[0], edge[1])

        net.save_graph("network.html")
        st.components.v1.html(open("network.html", "r").read(), height=600)

# -----------------------------
# Tab 3: Network Metrics
# -----------------------------
with tabs[2]:
    st.header("📊 Network Metrics")

    if df is not None:
        G = nx.Graph()
        for _, row in df.iterrows():
            try:
                p1, p2 = row["Protein1"], row["Protein2"]
                G.add_edge(p1, p2)
            except KeyError:
                continue

        closeness = nx.closeness_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        clustering = nx.clustering(G)

        metrics_df = pd.DataFrame({
            "Protein": list(G.nodes()),
            "Closeness": [closeness[n] for n in G.nodes()],
            "Betweenness": [betweenness[n] for n in G.nodes()],
            "ClusteringCoeff": [clustering[n] for n in G.nodes()]
        })

        st.dataframe(metrics_df, use_container_width=True)

# -----------------------------
# Tab 4: Protein Details
# -----------------------------
with tabs[3]:
    st.header("🧾 Protein Details")

    protein_id = st.text_input("Enter UniProt Protein ID", key="protein_input")
    if protein_id:
        url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.fasta"
        response = requests.get(url)

        if response.status_code == 200:
            fasta_text = response.text
            st.text_area("FASTA Sequence", fasta_text, height=200, key="fasta_output")
        else:
            st.error("Protein not found in UniProt.")

# -----------------------------
# Tab 5: Sequences
# -----------------------------
with tabs[4]:
    st.header("🧬 Protein Sequences")

    fasta_file = st.file_uploader("Upload a FASTA file", type=["fasta"], key="fasta_upload")
    if fasta_file:
        records = list(SeqIO.parse(fasta_file, "fasta"))
        for rec in records:
            st.text(f">{rec.id}\n{rec.seq}")
    else:
        st.info("No FASTA uploaded. Using sample.")
        records = list(SeqIO.parse(SAMPLE_FASTA_PATH, "fasta"))
        for rec in records:
            st.text(f">{rec.id}\n{rec.seq}")

# -----------------------------
# Tab 6: Motifs / Domains
# -----------------------------
with tabs[5]:
    st.header("📑 Motifs and Domains (UniProt API)")

    motif_protein = st.text_input("Enter UniProt ID for motifs/domains", key="motif_input")
    if motif_protein:
        url = f"https://rest.uniprot.org/uniprotkb/{motif_protein}.json"
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            features = data.get("features", [])
            for feat in features[:20]:
                st.write(f"**{feat.get('type','')}**: {feat.get('description','')} "
                         f"({feat.get('location',{})})")
        else:
            st.error("Could not fetch motifs/domains.")

# -----------------------------
# Tab 7: 3D Structure
# -----------------------------
with tabs[6]:
    st.header("🔬 3D Structure (AlphaFold)")

    pdb_id = st.text_input("Enter AlphaFold/UniProt ID", key="pdb_input")
    if pdb_id:
        viewer = py3Dmol.view(query=f"pdb:{pdb_id}")
        viewer.setStyle({"cartoon": {"color": "spectrum"}})
        viewer.zoomTo()
        viewer_html = viewer._make_html()
        st.components.v1.html(viewer_html, height=500, key="pdb_view")

# -----------------------------
# Tab 8: Intrinsic Disorder & Stability
# -----------------------------
with tabs[7]:
    st.header("🧩 Intrinsic Disorder & Stability")

    st.markdown("""
    ⚠️ This is a **placeholder** for your research.
    - You can integrate IUPred2A or DISOPRED APIs for intrinsic disorder.
    - Stability predictions can be integrated via external ML models.
    """)

    protein_disorder_input = st.text_input("Enter Protein ID for disorder analysis", key="disorder_input")
    if protein_disorder_input:
        st.info(f"Running intrinsic disorder & stability analysis for {protein_disorder_input} (to be implemented).")
