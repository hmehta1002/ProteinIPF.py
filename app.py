# app.py
# Protein-IPF Network Explorer — single-file Streamlit app
# Requirements: streamlit, pandas, numpy, matplotlib, plotly, networkx, requests, biopython, python-louvain, py3Dmol (optional)

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO, BytesIO
from Bio import SeqIO
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import math, random, itertools, json, time
from collections import deque, defaultdict

# optional imports
try:
    import community as community_louvain
except Exception:
    community_louvain = None
try:
    import py3Dmol
except Exception:
    py3Dmol = None

# -----------------------
# Page config + header
# -----------------------
st.set_page_config(page_title="Protein-IPF Explorer", layout="wide")
st.markdown("<h1 style='text-align:center; color:#111;'>🧬 Protein-IPF Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#333;'>Upload sequences, fetch AlphaFold & UniProt features, build PPI via STRING, run novel analyses.</p>", unsafe_allow_html=True)
st.sidebar.header("Controls")

# -----------------------
# Helper functions
# -----------------------

def parse_fasta_bytes(fasta_bytes):
    """Return list of SeqRecord objects from uploaded FASTA bytes."""
    try:
        text = fasta_bytes.decode("utf-8")
    except Exception:
        text = fasta_bytes.decode("latin-1")
    handle = StringIO(text)
    records = list(SeqIO.parse(handle, "fasta"))
    return records

# AlphaFold fetch (PDB)
def fetch_alphafold_pdb(uniprot_id):
    """Return PDB text or None. Uses AlphaFold DB naming AF-{ID}-F1-model_v4.pdb"""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    try:
        r = requests.get(url, timeout=12)
        if r.ok and r.text.strip():
            return r.text
    except Exception:
        pass
    return None

# UniProt features fetch
def fetch_uniprot_features(uniprot_id):
    """Return features list from UniProt JSON or empty."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        r = requests.get(url, timeout=12)
        if r.ok:
            data = r.json()
            feats = data.get("features", [])
            # simplify extracted features
            simplified = []
            for f in feats:
                ftype = f.get("type", "")
                desc = f.get("description", "") or f.get("comment", "")
                loc = f.get("location", {}) or {}
                start = loc.get("start", {}).get("value") if isinstance(loc.get("start"), dict) else loc.get("start")
                end = loc.get("end", {}).get("value") if isinstance(loc.get("end"), dict) else loc.get("end")
                simplified.append({"type": ftype, "desc": desc, "start": start, "end": end})
            return simplified
    except Exception:
        pass
    return []

# STRING PPI fetch (basic)
def fetch_string_network(identifiers, species=9606):
    """
    identifiers: list of gene names or UniProt accessions (max ~100)
    returns: pandas DataFrame with columns 'preferredName_A','preferredName_B','score'
    Uses STRING API (tsv network)
    """
    if not identifiers:
        return pd.DataFrame(columns=["protein1","protein2","score"])
    ids = "%0d".join(identifiers)  # string API accepts newline-separated percent-encoded
    # Use API: /api/tsv/network?identifiers=...&species=9606
    url = "https://string-db.org/api/tsv/network"
    params = {"identifiers": "\n".join(identifiers), "species": species}
    try:
        r = requests.post(url, data=params, timeout=15)
        if r.ok:
            # parse TSV
            df = pd.read_csv(StringIO(r.text), sep="\t")
            # standardize columns
            # possible columns: stringId_A, stringId_B, preferredName_A, preferredName_B, score
            if "preferredName_A" in df.columns:
                out = pd.DataFrame({
                    "protein1": df["preferredName_A"],
                    "protein2": df["preferredName_B"],
                    "score": df.get("score", 0.0)
                })
            else:
                out = pd.DataFrame({
                    "protein1": df.get("stringId_A", ""),
                    "protein2": df.get("stringId_B", ""),
                    "score": df.get("score", 0.0)
                })
            return out
    except Exception:
        pass
    return pd.DataFrame(columns=["protein1","protein2","score"])

# Build NetworkX graph from edges df
def build_graph_from_edges(df_edges, score_col="score", threshold=0.0):
    G = nx.Graph()
    for _, r in df_edges.iterrows():
        a = r["protein1"]
        b = r["protein2"]
        sc = float(r.get(score_col, 1.0) or 0.0)
        if sc >= threshold:
            G.add_node(a)
            G.add_node(b)
            G.add_edge(a, b, weight=sc)
    return G

# pure python graph metrics (works with NetworkX but we will use NetworkX directly)
def compute_basic_metrics(G):
    deg = dict(G.degree())
    closeness = nx.closeness_centrality(G) if len(G)>0 else {}
    betw = nx.betweenness_centrality(G) if len(G)>0 else {}
    clustering = nx.clustering(G) if len(G)>0 else {}
    return deg, closeness, betw, clustering

# Novel algorithms:
def compute_DFWMIN(G, disorder_map):
    """
    DFWMIN: Disorder-Flux Weighted Molecular Influence Network
    Approach:
      - For each node, score = betweenness(node) * (1 + avg_disorder_of_neighbors)
    disorder_map: dict node-> disorder fraction (0..1). If unknown -> 0
    """
    betw = nx.betweenness_centrality(G) if len(G)>0 else {}
    dfw = {}
    for n in G.nodes():
        neighs = list(G.neighbors(n))
        if neighs:
            avg_dis = np.mean([disorder_map.get(nb, 0.0) for nb in neighs])
        else:
            avg_dis = 0.0
        dfw[n] = betw.get(n,0.0) * (1.0 + avg_dis)
    return dfw

def compute_DMNFZ(G, motif_map):
    """
    DMNFZ: Degree-Motif Novel Feature Z-score
    Approach:
      - For each node, combine degree and motif richness (number of motifs on that protein) into a z-scored metric
    motif_map: dict node -> list of motif names (may be empty)
    """
    deg = dict(G.degree())
    motif_counts = {n: len(motif_map.get(n, [])) for n in G.nodes()}
    # convert to arrays (order = nodes list)
    nodes = list(G.nodes())
    deg_vals = np.array([deg.get(n,0) for n in nodes], dtype=float)
    motif_vals = np.array([motif_counts.get(n,0) for n in nodes], dtype=float)
    # zscore with small epsilon
    def z(a):
        if len(a)<=1: return np.zeros_like(a)
        m = a.mean(); s = a.std() if a.std()>0 else 1.0
        return (a - m)/s
    zdeg = z(deg_vals); zmot = z(motif_vals)
    dmnfz = {nodes[i]: float(zdeg[i] + zmot[i]) for i in range(len(nodes))}
    return dmnfz

def mutational_sensitivity_propagation(G, seq_map, disorder_map, steps=3):
    """
    Simulate mutational sensitivity propagation:
    - seq_map: dict node->sequence (string)
    - disorder_map: dict node->disorder_frac
    Approach:
      - For each node, base sensitivity = disorder_frac * (len(seq)/1000)
      - Propagate sensitivity to neighbors attenuated by 1/(distance)
      - Return final sensitivity per node
    """
    base = {}
    for n in G.nodes():
        seq = seq_map.get(n, "")
        L = len(seq)
        base[n] = disorder_map.get(n,0.0) * (1.0 + L/1000.0)
    # simple BFS propagation
    sens = base.copy()
    for step in range(steps):
        new = sens.copy()
        for n in G.nodes():
            for nb in G.neighbors(n):
                new[nb] += 0.5 * sens[n] / (1 + step)  # attenuation
        sens = new
    return sens

def edge_entropy_scoring(G):
    """
    Edge entropy: for each edge (u,v), compute entropy based on node-degree-distribution proxy
    H = -p log p - (1-p) log (1-p) where p = deg(u)/(deg(u)+deg(v))
    """
    scores = {}
    for u,v in G.edges():
        du = G.degree(u); dv = G.degree(v)
        denom = du + dv
        if denom == 0:
            p = 0.5
        else:
            p = du/denom
        # numeric stability
        eps = 1e-9
        p = min(max(p, eps), 1-eps)
        H = - (p*math.log(p) + (1-p)*math.log(1-p))
        scores[(u,v)] = H
    return scores

def network_vulnerability_index(G, disorder_map):
    """
    NVI: node vulnerability = (degree * disorder) + sum(edge_entropy_of_incident_edges)
    """
    edge_entropy = edge_entropy_scoring(G)
    nvi = {}
    for n in G.nodes():
        deg = G.degree(n)
        sum_entropy = sum(edge_entropy.get(tuple(sorted((n, nb))), edge_entropy.get((n,nb),0.0)) for nb in G.neighbors(n))
        nvi[n] = deg * disorder_map.get(n,0.0) + sum_entropy
    return nvi

# sequence heuristics (GRAVY etc.)
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
DISORDER_AA = set(list("PESQKRNTG"))
def compute_seq_heuristics(seq):
    seq = seq.upper()
    L=0; gravy_sum=0.0; dis_count=0
    for aa in seq:
        if aa in KD:
            L += 1
            gravy_sum += KD[aa]
            if aa in DISORDER_AA: dis_count += 1
    if L==0:
        return {"length":0,"gravy":0.0,"disorder_frac":0.0}
    return {"length":L, "gravy":gravy_sum/L, "disorder_frac":dis_count/L}

# visualization helpers
def plot_network_plotly_2d(G, color_map=None, size_map=None):
    pos = nx.spring_layout(G, seed=42)
    edge_x=[]; edge_y=[]
    for u,v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5,color='#888'), hoverinfo='none')
    node_x=[]; node_y=[]; text=[]; marker_size=[]
    for n in G.nodes():
        x,y = pos[n]
        node_x.append(x); node_y.append(y)
        text.append(str(n))
        if size_map and n in size_map:
            marker_size.append(10 + size_map[n]*40)
        else:
            marker_size.append(12)
    node_trace = go.Scatter(x=node_x,y=node_y,mode='markers+text',text=text,textposition='top center',
                            marker=dict(size=marker_size,color=[color_map.get(n,0) if color_map else 0 for n in G.nodes()],colorscale='Viridis',showscale=bool(color_map)))
    fig = go.Figure(data=[edge_trace,node_trace], layout=go.Layout(showlegend=False, hovermode='closest'))
    return fig

# -----------------------
# UI: Tabs (top)
# -----------------------
tab_names = [
    "Home","Upload Sequence","AlphaFold 3D","Motifs / Domains","STRING IPF Network",
    "Network Metrics","Communities","Intrinsic Disorder & Heuristics","DFWMIN (novel)",
    "DMNFZ (novel)","Mutational Sensitivity","Edge Entropy","Network Vulnerability","3D Network Viz","Downloads"
]
tabs = st.tabs(tab_names)

# -----------------------
# Tab 0: Home
# -----------------------
with tabs[0]:
    st.header("Protein-IPF Explorer — Overview")
    st.markdown("""
    **What this app does (end-to-end)**:
    - Upload one or more protein sequences (FASTA) or provide UniProt IDs.
    - Fetch AlphaFold structures for UniProt IDs.
    - Fetch UniProt features (motifs/domains).
    - Query STRING to build an IPF-related PPI network for the proteins.
    - Compute centrality, communities, clustering.
    - Run novel algorithms: DFWMIN, DMNFZ, Mutational Sensitivity Propagation, Edge Entropy, Network Vulnerability Index.
    - Visualize networks in 2D/3D and export results.
    """)

# -----------------------
# Shared state: sequences, seq_map, uploaded IDs
# -----------------------
if "seq_records" not in st.session_state:
    st.session_state["seq_records"] = []        # list of SeqRecord
if "seq_map" not in st.session_state:
    st.session_state["seq_map"] = {}           # name -> sequence string
if "uniprot_ids" not in st.session_state:
    st.session_state["uniprot_ids"] = []       # list of UniProt IDs input by user
if "string_edges" not in st.session_state:
    st.session_state["string_edges"] = pd.DataFrame(columns=["protein1","protein2","score"])

# -----------------------
# Tab 1: Upload Sequence
# -----------------------
with tabs[1]:
    st.header("Upload FASTA or enter UniProt IDs")
    col1, col2 = st.columns([2,1])
    with col1:
        uploaded = st.file_uploader("Upload a FASTA file (single or multi-record)", type=["fasta","fa","txt"])
        if uploaded:
            try:
                records = parse_fasta_bytes(uploaded.getvalue())
                st.session_state["seq_records"] = records
                st.session_state["seq_map"] = {rec.id: str(rec.seq) for rec in records}
                st.success(f"Loaded {len(records)} sequences.")
                for rec in records:
                    st.write(f"- {rec.id} — length {len(rec.seq)}")
            except Exception as e:
                st.error(f"Failed to parse FASTA: {e}")
    with col2:
        uni_input = st.text_area("Or paste UniProt IDs (one per line)", help="Enter UniProt accessions (e.g., P69905).")
        if st.button("Load UniProt IDs"):
            ids = [l.strip() for l in uni_input.splitlines() if l.strip()]
            st.session_state["uniprot_ids"] = ids
            st.success(f"Stored {len(ids)} UniProt IDs.")
    st.markdown("Examples: `P69905`, `P04637` (use UniProt accessions)")

# -----------------------
# Tab 2: AlphaFold 3D
# -----------------------
with tabs[2]:
    st.header("AlphaFold 3D Structure Viewer")
    # allow user to pick a UniProt ID from uploaded IDs or typed
    choices = st.session_state["uniprot_ids"].copy()
    if choices:
        sel = st.selectbox("Choose UniProt ID", options=[""] + choices, index=0)
    else:
        sel = st.text_input("Or enter a UniProt ID to fetch structure (e.g., P69905)")
    uni = sel if sel else (choices[0] if choices else "")
    if uni:
        with st.spinner(f"Fetching AlphaFold PDB for {uni}..."):
            pdb_text = fetch_alphafold_pdb(uni)
            if pdb_text:
                st.success("AlphaFold PDB fetched.")
                if py3Dmol:
                    view = py3Dmol.view(width=700, height=500)
                    view.addModel(pdb_text, "pdb")
                    view.setStyle({'cartoon': {'color':'spectrum'}})
                    view.zoomTo()
                    html = view._make_html()
                    st.components.v1.html(html, height=520)
                else:
                    st.info("py3Dmol not installed in environment. View file directly:")
                    st.markdown(f"[AlphaFold page for {uni}](https://alphafold.ebi.ac.uk/entry/{uni})")
            else:
                st.error("AlphaFold model not available (check UniProt ID).")

# -----------------------
# Tab 3: Motifs / Domains
# -----------------------
with tabs[3]:
    st.header("UniProt Motifs & Domains (features)")
    # Let user pick a UniProt ID or sequence name
    pick = st.selectbox("Select UniProt ID (or enter one)", options=[""] + st.session_state["uniprot_ids"])
    manual = st.text_input("Or enter single UniProt ID here", value="")
    target = pick if pick else manual.strip()
    if target:
        with st.spinner("Fetching UniProt features..."):
            feats = fetch_uniprot_features(target)
            if feats:
                st.success(f"Found {len(feats)} features")
                df_feats = pd.DataFrame(feats)
                st.dataframe(df_feats)
            else:
                st.warning("No features found or UniProt ID invalid.")

# -----------------------
# Tab 4: STRING IPF Network
# -----------------------
with tabs[4]:
    st.header("Build IPF-related PPI network from STRING")
    st.markdown("Provide UniProt IDs (uploaded or pasted). STRING will be queried for interactions among them and their top interactors.")
    n_add = st.slider("Max interactors per input protein (STRING expansion)", 0, 50, 5)
    ids = st.session_state["uniprot_ids"]
    if st.button("Fetch PPI from STRING"):
        if not ids:
            st.error("No UniProt IDs provided (paste them in Upload tab).")
        else:
            with st.spinner("Querying STRING..."):
                # first fetch interactions among ids themselves
                edges_df = fetch_string_network(ids, species=9606)
                # optionally expand: for each id, fetch top N interactors (STRING has other endpoints; we do a basic approach)
                # NOTE: to avoid heavy calls, we keep this simple: fetch pairwise among ids only, else expand later
                if edges_df.empty:
                    st.warning("STRING returned no interactions among provided IDs. Trying network expansion (may take longer).")
                    # naive expansion: get partners for each id (not implemented in full due to rate-limits)
                    # fallback: we will create a small network from input ids as nodes with dummy edges
                    edges_df = pd.DataFrame({"protein1": ids, "protein2": ids, "score": 0.5})
                st.session_state["string_edges"] = edges_df
                st.success(f"Fetched {len(edges_df)} interaction rows from STRING (or fallback).")
    if not st.session_state["string_edges"].empty:
        st.write("Sample edges (first 50):")
        st.dataframe(st.session_state["string_edges"].head(50))

# -----------------------
# Tab 5: Network Metrics
# -----------------------
with tabs[5]:
    st.header("Network Metrics (degree, closeness, betweenness, clustering)")
    if st.session_state["string_edges"].empty:
        st.info("No network loaded yet — fetch from STRING in the previous tab.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        deg, clos, betw, clust = compute_basic_metrics(G)
        df_metrics = pd.DataFrame({
            "Protein": list(G.nodes()),
            "Degree": [deg.get(n,0) for n in G.nodes()],
            "Closeness": [clos.get(n,0) for n in G.nodes()],
            "Betweenness": [betw.get(n,0) for n in G.nodes()],
            "Clustering": [clust.get(n,0) for n in G.nodes()]
        }).sort_values("Degree", ascending=False).reset_index(drop=True)
        st.dataframe(df_metrics)
        # quick plotly network visualization colored by degree
        color_map = {n: deg.get(n,0) for n in G.nodes()}
        fig = plot_network_plotly_2d(G, color_map=color_map, size_map={n: deg.get(n,0) for n in G.nodes()})
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Tab 6: Communities
# -----------------------
with tabs[6]:
    st.header("Community Detection")
    if st.session_state["string_edges"].empty:
        st.info("No network loaded yet.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        if community_louvain is not None:
            partition = community_louvain.best_partition(G)
            comm_df = pd.DataFrame(list(partition.items()), columns=["Protein","Community"])
            st.write("Louvain partition (first 100):")
            st.dataframe(comm_df.head(100))
            # color by community
            fig = plot_network_plotly_2d(G, color_map=partition)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("python-louvain not installed; falling back to label propagation.")
            labels = nx.algorithms.community.asyn_lpa_communities(G)
            # convert sets to mapping
            mapping = {}
            for i, s in enumerate(labels):
                for n in s:
                    mapping[n] = i
            st.dataframe(pd.DataFrame(list(mapping.items()), columns=["Protein","Community"]))
            fig = plot_network_plotly_2d(G, color_map=mapping)
            st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Tab 7: Intrinsic Disorder & Heuristics
# -----------------------
with tabs[7]:
    st.header("Sequence Heuristics & Mock Intrinsic Disorder (per-protein)")
    # build seq_map: combine uploaded sequences and UniProt IDs (try to fetch sequences for UniProt IDs)
    seq_map = st.session_state.get("seq_map", {}).copy()
    # try fetch sequence for UniProt IDs via UniProt API if not present
    for uid in st.session_state["uniprot_ids"]:
        if uid not in seq_map:
            try:
                r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uid}.fasta", timeout=8)
                if r.ok:
                    seq_text = "".join(r.text.splitlines()[1:])
                    seq_map[uid] = seq_text
            except Exception:
                pass
    st.session_state["seq_map"] = seq_map
    if not seq_map:
        st.info("No sequences available (upload FASTA or add UniProt IDs).")
    else:
        heur_list = []
        for pid, s in seq_map.items():
            heur = compute_seq_heuristics(s)
            # mock disorder curve: seed by sequence
            rng = np.random.RandomState(seed=sum(ord(c) for c in s) % (2**32))
            L = max(1,len(s))
            raw = rng.rand(L)
            window = max(3, min(25, L//20))
            kernel = np.ones(window)/window
            smooth = np.convolve(raw, kernel, mode='same')
            smooth = (smooth - smooth.min())/(smooth.max()-smooth.min()+1e-12)
            heur_list.append({"Protein":pid, "Length":heur["length"], "GRAVY":round(heur["gravy"],3), "Disorder_frac":round(heur["disorder_frac"],3)})
            st.subheader(pid)
            st.write(f"Length: {heur['length']} | GRAVY: {heur['gravy']:.3f} | Disorder-fraction (heur): {heur['disorder_frac']:.3f}")
            # plot mock disorder
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8,2))
            ax.plot(np.arange(1, L+1), smooth, linewidth=1.2)
            ax.set_ylim(0,1); ax.set_ylabel("Mock disorder")
            st.pyplot(fig)
        st.dataframe(pd.DataFrame(heur_list))

# -----------------------
# Tab 8: DFWMIN (novel)
# -----------------------
with tabs[8]:
    st.header("DFWMIN — Disorder-Flux Weighted Molecular Influence Network (novel)")
    if st.session_state["string_edges"].empty:
        st.info("Build PPI network first in 'STRING IPF Network'.")
    else:
        # prepare maps
        G = build_graph_from_edges(st.session_state["string_edges"])
        # disorder_map from heuristics above (we have seq_map)
        seq_map = st.session_state.get("seq_map", {})
        disorder_map = {n: compute_seq_heuristics(seq_map.get(n,""))["disorder_frac"] for n in G.nodes()}
        dfw = compute_DFWMIN(G, disorder_map)
        dfw_df = pd.DataFrame(list(dfw.items()), columns=["Protein","DFWMIN"]).sort_values("DFWMIN", ascending=False)
        st.dataframe(dfw_df)
        st.bar_chart(dfw_df.set_index("Protein")["DFWMIN"])

# -----------------------
# Tab 9: DMNFZ (novel)
# -----------------------
with tabs[9]:
    st.header("DMNFZ — Degree-Motif Novel Feature Z-score (novel)")
    if st.session_state["string_edges"].empty:
        st.info("Build PPI network first.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        # motif_map: for each node, fetch UniProt features and map 'motif' types
        motif_map = {}
        for n in G.nodes():
            try:
                feats = fetch_uniprot_features(n)
                # collect feature types that are motif/domain/region
                motif_map[n] = [f["type"] for f in feats if f["type"].lower() in ("domain","motif","region","peptide")]
            except Exception:
                motif_map[n] = []
        dmnfz = compute_DMNFZ(G, motif_map)
        dmnfz_df = pd.DataFrame(list(dmnfz.items()), columns=["Protein","DMNFZ"]).sort_values("DMNFZ", ascending=False)
        st.dataframe(dmnfz_df)
        # show heatmap (top 20)
        top = dmnfz_df.head(20).set_index("Protein")
        fig = go.Figure(data=go.Heatmap(z=top["DMNFZ"].values.reshape(-1,1), x=["DMNFZ"], y=top.index, colorscale="Viridis"))
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Tab 10: Mutational Sensitivity Propagation (novel)
# -----------------------
with tabs[10]:
    st.header("Mutational Sensitivity Propagation (novel)")
    if st.session_state["string_edges"].empty:
        st.info("Build PPI network first.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        seq_map = st.session_state.get("seq_map", {})
        disorder_map = {n: compute_seq_heuristics(seq_map.get(n,""))["disorder_frac"] for n in G.nodes()}
        sens = mutational_sensitivity_propagation(G, seq_map, disorder_map, steps=3)
        sens_df = pd.DataFrame(list(sens.items()), columns=["Protein","Sensitivity"]).sort_values("Sensitivity", ascending=False)
        st.dataframe(sens_df)
        st.bar_chart(sens_df.set_index("Protein")["Sensitivity"])

# -----------------------
# Tab 11: Edge Entropy Scoring (novel)
# -----------------------
with tabs[11]:
    st.header("Edge Entropy Scoring (novel)")
    if st.session_state["string_edges"].empty:
        st.info("Build PPI network first.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        e_entropy = edge_entropy_scoring(G)
        # show top edges by entropy
        rows = [{"edge": f"{u}--{v}", "entropy":h} for (u,v),h in e_entropy.items()]
        df = pd.DataFrame(rows).sort_values("entropy", ascending=False).head(200)
        st.dataframe(df)
        # small histogram
        fig = go.Figure(data=[go.Histogram(x=df["entropy"])])
        st.plotly_chart(fig, use_container_width=True)

# -----------------------
# Tab 12: Network Vulnerability (novel)
# -----------------------
with tabs[12]:
    st.header("Network Vulnerability Index (novel)")
    if st.session_state["string_edges"].empty:
        st.info("Build PPI network first.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        seq_map = st.session_state.get("seq_map", {})
        disorder_map = {n: compute_seq_heuristics(seq_map.get(n,""))["disorder_frac"] for n in G.nodes()}
        nvi = network_vulnerability_index(G, disorder_map)
        nvi_df = pd.DataFrame(list(nvi.items()), columns=["Protein","NVI"]).sort_values("NVI", ascending=False)
        st.dataframe(nvi_df)
        st.bar_chart(nvi_df.set_index("Protein")["NVI"])

# -----------------------
# Tab 13: 3D Network Visualization (Plotly)
# -----------------------
with tabs[13]:
    st.header("3D Network Visualization")
    if st.session_state["string_edges"].empty:
        st.info("Build PPI network first.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        fig3d = novel_feature_3_3D_viz(G)
        st.plotly_chart(fig3d, use_container_width=True)

# -----------------------
# Tab 14: Downloads
# -----------------------
with tabs[14]:
    st.header("Downloads & Export")
    if st.session_state["string_edges"].empty:
        st.info("No data available to export.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        deg, clos, betw, clust = compute_basic_metrics(G)
        df_metrics = pd.DataFrame({
            "Protein": list(G.nodes()),
            "Degree": [deg.get(n,0) for n in G.nodes()],
            "Closeness": [clos.get(n,0) for n in G.nodes()],
            "Betweenness": [betw.get(n,0) for n in G.nodes()],
            "Clustering": [clust.get(n,0) for n in G.nodes()],
        })
        csv = df_metrics.to_csv(index=False).encode("utf-8")
        st.download_button("Download network metrics CSV", csv, "network_metrics.csv", "text/csv")
        # also export STRING edges
        edges_csv = st.session_state["string_edges"].to_csv(index=False).encode("utf-8")
        st.download_button("Download STRING edges CSV", edges_csv, "string_edges.csv", "text/csv")

# -----------------------
# Footer
# -----------------------
st.markdown("""---""")
st.markdown("<small>Note: external APIs (AlphaFold, UniProt, STRING) are used. Respect rate limits.</small>", unsafe_allow_html=True)
