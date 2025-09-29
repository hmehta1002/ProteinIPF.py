# app.py
# Protein-IPF Network Explorer — complete, single-file Streamlit app
# Minimal requirements listed below. Optional: py3Dmol for inline 3D.

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from Bio import SeqIO
import networkx as nx
import plotly.graph_objs as go
import math, random, json, time
from collections import defaultdict

# Optional imports (graceful fallback)
try:
    import community as community_louvain
except Exception:
    community_louvain = None
try:
    import py3Dmol
except Exception:
    py3Dmol = None

# Page config
st.set_page_config(page_title="Protein-IPF Explorer", layout="wide")
st.markdown("<h1 style='text-align:center;'>🧬 Protein-IPF Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Upload sequences, fetch AlphaFold & UniProt, build PPI via STRING, run novel analyses.</p>", unsafe_allow_html=True)

# -----------------------------
# Helpers (parsing, API fetch, graph)
# -----------------------------

def parse_fasta_bytes(fasta_bytes):
    """Return list of SeqRecord from FASTA bytes (supports multi-record)."""
    try:
        text = fasta_bytes.decode("utf-8")
    except Exception:
        text = fasta_bytes.decode("latin-1")
    handle = StringIO(text)
    records = list(SeqIO.parse(handle, "fasta"))
    return records

@st.cache_data(show_spinner=False)
def fetch_alphafold_pdb(uniprot_id):
    """Fetch AlphaFold PDB text for UniProt ID (returns None if not found)."""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    try:
        r = requests.get(url, timeout=12)
        if r.ok and r.text.strip():
            return r.text
    except Exception:
        pass
    return None

@st.cache_data(show_spinner=False)
def fetch_uniprot_features(uniprot_id):
    """Fetch UniProt features via REST API; returns list of simplified feature dicts."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        r = requests.get(url, timeout=12)
        if r.ok:
            data = r.json()
            feats = data.get("features", [])
            simplified = []
            for f in feats:
                ftype = f.get("type", "")
                desc = f.get("description", "") or f.get("comment", "")
                loc = f.get("location", {}) or {}
                # start/end handling resilient to structure
                start = None; end = None
                s = loc.get("start")
                e = loc.get("end")
                if isinstance(s, dict):
                    start = s.get("value")
                else:
                    start = s
                if isinstance(e, dict):
                    end = e.get("value")
                else:
                    end = e
                simplified.append({"type": ftype, "desc": desc, "start": start, "end": end, "raw": f})
            return simplified
    except Exception:
        pass
    return []

@st.cache_data(show_spinner=False)
def fetch_string_network(identifiers, species=9606):
    """
    Query STRING for interactions among identifiers.
    Returns a DataFrame with columns: protein1, protein2, score
    Note: identifiers should be list of UniProt accessions or gene symbols.
    """
    if not identifiers:
        return pd.DataFrame(columns=["protein1","protein2","score"])
    url = "https://string-db.org/api/tsv/network"
    params = {"identifiers": "\n".join(identifiers), "species": species}
    try:
        r = requests.post(url, data=params, timeout=20)
        if r.ok and r.text.strip():
            df = pd.read_csv(StringIO(r.text), sep="\t")
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

def build_graph_from_edges(df_edges, score_col="score", threshold=0.0):
    G = nx.Graph()
    for _, r in df_edges.iterrows():
        a = str(r["protein1"])
        b = str(r["protein2"])
        try:
            sc = float(r.get(score_col, 0.0) or 0.0)
        except Exception:
            sc = 0.0
        if sc >= threshold:
            G.add_node(a)
            G.add_node(b)
            G.add_edge(a, b, weight=sc)
    return G

def compute_basic_metrics(G):
    deg = dict(G.degree())
    closeness = nx.closeness_centrality(G) if len(G)>0 else {}
    betw = nx.betweenness_centrality(G) if len(G)>0 else {}
    clustering = nx.clustering(G) if len(G)>0 else {}
    return deg, closeness, betw, clustering

# Sequence heuristics
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
DISORDER_AA = set(list("PESQKRNTG"))
def compute_seq_heuristics(seq):
    seq = (seq or "").upper()
    L=0; gravy_sum=0.0; dis_count=0
    for aa in seq:
        if aa in KD:
            L += 1
            gravy_sum += KD[aa]
            if aa in DISORDER_AA: dis_count += 1
    if L==0:
        return {"length":0,"gravy":0.0,"disorder_frac":0.0}
    return {"length":L, "gravy":gravy_sum/L, "disorder_frac":dis_count/L}

# Novel algorithms (as before) ------------------------------------------------
def compute_DFWMIN(G, disorder_map):
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
    deg = dict(G.degree())
    motif_counts = {n: len(motif_map.get(n, [])) for n in G.nodes()}
    nodes = list(G.nodes())
    deg_vals = np.array([deg.get(n,0) for n in nodes], dtype=float)
    motif_vals = np.array([motif_counts.get(n,0) for n in nodes], dtype=float)
    def z(a):
        if len(a)<=1: return np.zeros_like(a)
        m = a.mean(); s = a.std() if a.std()>0 else 1.0
        return (a - m)/s
    zdeg = z(deg_vals); zmot = z(motif_vals)
    dmnfz = {nodes[i]: float(zdeg[i] + zmot[i]) for i in range(len(nodes))}
    return dmnfz

def mutational_sensitivity_propagation(G, seq_map, disorder_map, steps=3):
    base = {}
    for n in G.nodes():
        seq = seq_map.get(n, "")
        L = len(seq)
        base[n] = disorder_map.get(n,0.0) * (1.0 + L/1000.0)
    sens = base.copy()
    for step in range(steps):
        new = sens.copy()
        for n in G.nodes():
            for nb in G.neighbors(n):
                new[nb] += 0.5 * sens[n] / (1 + step)
        sens = new
    return sens

def edge_entropy_scoring(G):
    scores = {}
    for u,v in G.edges():
        du = G.degree(u); dv = G.degree(v)
        denom = du + dv
        p = 0.5 if denom == 0 else du/denom
        eps = 1e-9
        p = min(max(p, eps), 1-eps)
        H = - (p*math.log(p) + (1-p)*math.log(1-p))
        scores[(u,v)] = H
    return scores

def network_vulnerability_index(G, disorder_map):
    edge_entropy = edge_entropy_scoring(G)
    nvi = {}
    for n in G.nodes():
        deg = G.degree(n)
        sum_entropy = 0.0
        for nb in G.neighbors(n):
            key1 = (n, nb)
            key2 = (nb, n)
            sum_entropy += edge_entropy.get(key1, edge_entropy.get(key2, 0.0))
        nvi[n] = deg * disorder_map.get(n,0.0) + sum_entropy
    return nvi

# Visualization helpers -------------------------------------------------------
def plot_network_plotly_2d(G, color_map=None, size_map=None):
    pos = nx.spring_layout(G, seed=42)
    edge_x=[]; edge_y=[]
    for u,v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.8,color='#888'), hoverinfo='none')
    node_x=[]; node_y=[]; text=[]; marker_size=[]; marker_color=[]
    for n in G.nodes():
        x,y = pos[n]
        node_x.append(x); node_y.append(y)
        text.append(str(n))
        marker_size.append(10 + (size_map.get(n,0)*40 if size_map else 0))
        marker_color.append(color_map.get(n,0) if color_map else 0)
    node_trace = go.Scatter(x=node_x,y=node_y,mode='markers+text',text=text,textposition='top center',
                            marker=dict(size=marker_size,color=marker_color,colorscale='Viridis',showscale=bool(color_map)))
    fig = go.Figure(data=[edge_trace,node_trace], layout=go.Layout(showlegend=False, hovermode='closest'))
    return fig

def novel_feature_3_3D_viz(G):
    pos = nx.spring_layout(G, dim=3, seed=42)
    edge_trace = go.Scatter3d(x=[], y=[], z=[], mode='lines', line=dict(width=2,color='blue'), hoverinfo='none')
    for u,v in G.edges():
        x0,y0,z0 = pos[u]; x1,y1,z1 = pos[v]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)
        edge_trace['z'] += (z0, z1, None)
    node_trace = go.Scatter3d(x=[], y=[], z=[], mode='markers+text', marker=dict(size=6,color='red'), text=list(G.nodes()))
    for n in G.nodes():
        x,y,z = pos[n]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['z'] += (z,)
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(scene=dict(xaxis=dict(showbackground=False), yaxis=dict(showbackground=False), zaxis=dict(showbackground=False)))
    return fig

# -----------------------------
# App state (session)
# -----------------------------
if "seq_records" not in st.session_state:
    st.session_state["seq_records"] = []
if "seq_map" not in st.session_state:
    st.session_state["seq_map"] = {}
if "uniprot_ids" not in st.session_state:
    st.session_state["uniprot_ids"] = []
if "string_edges" not in st.session_state:
    st.session_state["string_edges"] = pd.DataFrame(columns=["protein1","protein2","score"])

# -----------------------------
# Top tabs
# -----------------------------
tab_names = [
    "Home","Upload Sequence","AlphaFold 3D","Motifs / Domains","STRING IPF Network",
    "Network Metrics","Communities","Intrinsic Disorder & Heuristics","DFWMIN (novel)",
    "DMNFZ (novel)","Mutational Sensitivity","Edge Entropy","Network Vulnerability","3D Network Viz","Downloads"
]
tabs = st.tabs(tab_names)

# Tab 0: Home
with tabs[0]:
    st.header("Protein-IPF Explorer")
    st.write("Upload sequences or UniProt IDs, fetch AlphaFold / UniProt features, build STRING PPI and run novel analyses.")

# Tab 1: Upload
with tabs[1]:
    st.header("Upload FASTA or paste UniProt IDs")
    c1, c2 = st.columns([2,1])
    with c1:
        uploaded = st.file_uploader("Upload FASTA (single or multi-record)", type=["fasta","fa","txt"])
        if uploaded:
            try:
                records = parse_fasta_bytes(uploaded.getvalue())
                st.session_state["seq_records"] = records
                st.session_state["seq_map"] = {rec.id: str(rec.seq) for rec in records}
                st.success(f"Loaded {len(records)} sequences.")
                for rec in records:
                    st.write(f"- {rec.id} (len {len(rec.seq)})")
            except Exception as e:
                st.error(f"Failed parsing FASTA: {e}")
    with c2:
        uni_text = st.text_area("Paste UniProt IDs (one per line)")
        if st.button("Load UniProt IDs"):
            ids = [l.strip() for l in uni_text.splitlines() if l.strip()]
            st.session_state["uniprot_ids"] = ids
            st.info(f"Stored {len(ids)} UniProt IDs")

    st.markdown("Examples: `P69905`, `P04637`")

# Tab 2: AlphaFold 3D
with tabs[2]:
    st.header("AlphaFold 3D Viewer")
    choices = st.session_state["uniprot_ids"]
    sel = st.selectbox("Choose UniProt ID", options=[""] + choices) if choices else st.text_input("Or enter UniProt ID")
    uni = sel if sel else None
    manual = st.text_input("Manual UniProt ID (optional - overrides)")
    if manual.strip():
        uni = manual.strip()
    if uni:
        with st.spinner("Fetching AlphaFold model..."):
            pdb = fetch_alphafold_pdb(uni)
            if pdb:
                st.success("Fetched PDB")
                if py3Dmol:
                    view = py3Dmol.view(width=700, height=500)
                    view.addModel(pdb, "pdb")
                    view.setStyle({'cartoon': {'color':'spectrum'}})
                    view.zoomTo()
                    st.components.v1.html(view._make_html(), height=520)
                else:
                    st.markdown(f"py3Dmol not installed — open on AlphaFold site: https://alphafold.ebi.ac.uk/entry/{uni}")
            else:
                st.error("AlphaFold model not available for this ID.")

# Tab 3: Motifs / Domains
with tabs[3]:
    st.header("UniProt Motifs & Domains")
    pick = st.selectbox("Select UniProt ID (or enter)", options=[""] + st.session_state["uniprot_ids"])
    manual_in = st.text_input("Or enter UniProt ID here")
    target = pick if pick else manual_in.strip()
    if target:
        with st.spinner("Fetching UniProt features..."):
            feats = fetch_uniprot_features(target)
            if feats:
                df_feats = pd.DataFrame(feats)
                # filter typical types and show featureId if present in raw
                def get_feat_id(raw):
                    return raw.get("featureId") if isinstance(raw, dict) else None
                df_feats["feature_id"] = df_feats["raw"].apply(lambda r: r.get("featureId") if isinstance(r, dict) else None)
                st.dataframe(df_feats[["type","desc","start","end","feature_id"]])
            else:
                st.warning("No features found or invalid UniProt ID.")

# Tab 4: STRING IPF Network
with tabs[4]:
    st.header("Build IPF-related PPI network (STRING)")
    st.write("Provide UniProt IDs (uploaded or pasted) and click Fetch. You can also upload a text/csv file of IDs.")
    ids_file = st.file_uploader("Optional: upload file with UniProt IDs (txt/csv)", type=["txt","csv"])
    uploaded_ids = []
    if ids_file:
        try:
            if ids_file.name.endswith(".csv"):
                df_ids = pd.read_csv(ids_file)
                uploaded_ids = df_ids.iloc[:,0].astype(str).tolist()
            else:
                uploaded_ids = ids_file.getvalue().decode().splitlines()
            uploaded_ids = [x.strip() for x in uploaded_ids if x.strip()]
            st.info(f"Loaded {len(uploaded_ids)} IDs from file.")
        except Exception as e:
            st.error(f"Could not read IDs file: {e}")
    manual_ids = st.text_area("Or paste UniProt IDs (newline/comma separated)")
    manual_list = [x.strip() for x in manual_ids.replace(",", "\n").splitlines() if x.strip()]
    ids = list(dict.fromkeys(uploaded_ids + manual_list + st.session_state["uniprot_ids"]))
    st.write(f"Identifiers to query: {len(ids)} (showing up to 50):")
    st.write(ids[:50])
    expand_n = st.slider("STRING expansion: top X interactors per query", min_value=0, max_value=50, value=0)
    if st.button("Fetch PPI from STRING"):
        if not ids:
            st.error("No IDs provided.")
        else:
            with st.spinner("Querying STRING (may take a few seconds)..."):
                try:
                    edges_df = fetch_string_network(ids, species=9606)
                    # If expansion requested, note: robust expansion would require per-id queries (rate-limit heavy).
                    # For now we use pairwise among provided ids; advanced expansion could be added.
                    if edges_df.empty:
                        st.warning("No interactions returned by STRING for these identifiers.")
                    else:
                        st.session_state["string_edges"] = edges_df
                        st.success(f"Fetched {len(edges_df)} interactions from STRING.")
                except Exception as e:
                    st.error(f"STRING fetch failed: {e}")
    if not st.session_state["string_edges"].empty:
        st.dataframe(st.session_state["string_edges"].head(200))

# Tab 5: Network Metrics
with tabs[5]:
    st.header("Network Metrics")
    if st.session_state["string_edges"].empty:
        st.info("No network loaded. Build it in the STRING tab first.")
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
        fig = plot_network_plotly_2d(G, color_map=deg, size_map=deg)
        st.plotly_chart(fig, use_container_width=True)

# Tab 6: Communities
with tabs[6]:
    st.header("Community Detection")
    if st.session_state["string_edges"].empty:
        st.info("No network loaded.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        if community_louvain is not None:
            partition = community_louvain.best_partition(G)
            comm_df = pd.DataFrame(list(partition.items()), columns=["Protein","Community"])
            st.dataframe(comm_df)
            st.plotly_chart(plot_network_plotly_2d(G, color_map=partition), use_container_width=True)
        else:
            st.warning("python-louvain not installed; using label propagation fallback.")
            labs = nx.algorithms.community.asyn_lpa_communities(G)
            mapping = {}
            for i, s in enumerate(labs):
                for n in s:
                    mapping[n] = i
            st.dataframe(pd.DataFrame(list(mapping.items()), columns=["Protein","Community"]))
            st.plotly_chart(plot_network_plotly_2d(G, color_map=mapping), use_container_width=True)

# Tab 7: Intrinsic Disorder & Heuristics (fixed)
with tabs[7]:
    st.header("Intrinsic Disorder & Sequence Heuristics (per-protein)")
    # Ensure seq_map contains uploaded sequences and UniProt-fetched sequences
    seq_map = st.session_state.get("seq_map", {}).copy()
    # fetch sequences for UniProt IDs if missing
    for uid in st.session_state.get("uniprot_ids", []):
        if uid not in seq_map:
            try:
                r = requests.get(f"https://rest.uniprot.org/uniprotkb/{uid}.fasta", timeout=10)
                if r.ok:
                    seq_text = "".join(r.text.splitlines()[1:])
                    seq_map[uid] = seq_text
            except Exception:
                pass
    st.session_state["seq_map"] = seq_map

    if not seq_map:
        st.info("No sequences available. Upload FASTA or add UniProt IDs in Upload tab.")
    else:
        st.write("Computed heuristics (GRAVY, disorder-promotion fraction) and mock intrinsic disorder curve.")
        rows = []
        for pid, seq in seq_map.items():
            heur = compute_seq_heuristics(seq)
            rows.append({"Protein": pid, "Length": heur["length"], "GRAVY": round(heur["gravy"],4), "Disorder_frac": round(heur["disorder_frac"],4)})
        dfh = pd.DataFrame(rows).sort_values("Length", ascending=False).reset_index(drop=True)
        st.dataframe(dfh)

        # allow selecting one protein to show full mock disorder curve
        sel = st.selectbox("Select protein to plot mock disorder curve", options=list(seq_map.keys()))
        if sel:
            seq = seq_map[sel]
            L = max(1, len(seq))
            # deterministic random seed per sequence for reproducibility
            rng = np.random.RandomState(seed=sum(ord(c) for c in seq) % (2**32))
            raw = rng.rand(L)
            window = max(3, min(25, max(3, L//20)))
            kernel = np.ones(window)/window
            smooth = np.convolve(raw, kernel, mode='same')
            smooth = (smooth - smooth.min())/(smooth.max()-smooth.min()+1e-12)
            # Plot with Plotly for reliability
            trace = go.Scatter(x=list(range(1, L+1)), y=smooth, mode='lines', name='Mock disorder')
            layout = go.Layout(title=f"Mock intrinsic disorder for {sel} (length {L})",
                               xaxis=dict(title='Residue position'), yaxis=dict(title='Disorder score', range=[0,1]))
            fig = go.Figure(data=[trace], layout=layout)
            st.plotly_chart(fig, use_container_width=True)

# Tab 8: DFWMIN
with tabs[8]:
    st.header("DFWMIN (Disorder-Flux Weighted Molecular Influence Network)")
    if st.session_state["string_edges"].empty:
        st.info("Build PPI network in STRING tab first.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        seq_map = st.session_state.get("seq_map", {})
        disorder_map = {n: compute_seq_heuristics(seq_map.get(n,""))["disorder_frac"] for n in G.nodes()}
        dfw = compute_DFWMIN(G, disorder_map)
        dfw_df = pd.DataFrame(list(dfw.items()), columns=["Protein","DFWMIN"]).sort_values("DFWMIN", ascending=False)
        st.dataframe(dfw_df)
        st.bar_chart(dfw_df.set_index("Protein")["DFWMIN"])

# Tab 9: DMNFZ
with tabs[9]:
    st.header("DMNFZ (Degree-Motif Novel Feature Z-score)")
    if st.session_state["string_edges"].empty:
        st.info("Build PPI network in STRING tab first.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        # motif_map per node: query UniProt features (cached)
        motif_map = {}
        for n in list(G.nodes()):
            try:
                feats = fetch_uniprot_features(n)
                motif_map[n] = [f["type"] for f in feats if str(f.get("type","")).lower() in ("motif","domain","region","repeat","peptide")]
            except Exception:
                motif_map[n] = []
        dmnfz = compute_DMNFZ(G, motif_map)
        dmnfz_df = pd.DataFrame(list(dmnfz.items()), columns=["Protein","DMNFZ"]).sort_values("DMNFZ", ascending=False)
        st.dataframe(dmnfz_df)
        # small heatmap top 20
        top = dmnfz_df.head(20).set_index("Protein")
        if not top.empty:
            fig = go.Figure(data=go.Heatmap(z=top["DMNFZ"].values.reshape(-1,1), x=["DMNFZ"], y=top.index, colorscale="Viridis"))
            st.plotly_chart(fig, use_container_width=True)

# Tab 10: Mutational Sensitivity
with tabs[10]:
    st.header("Mutational Sensitivity Propagation")
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

# Tab 11: Edge Entropy
with tabs[11]:
    st.header("Edge Entropy Scoring")
    if st.session_state["string_edges"].empty:
        st.info("Build PPI network first.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        e_entropy = edge_entropy_scoring(G)
        rows = [{"edge": f"{u}--{v}", "entropy":h} for (u,v),h in e_entropy.items()]
        df = pd.DataFrame(rows).sort_values("entropy", ascending=False).reset_index(drop=True)
        st.dataframe(df.head(500))
        if not df.empty:
            fig = go.Figure(data=[go.Histogram(x=df["entropy"], nbinsx=30)])
            st.plotly_chart(fig, use_container_width=True)

# Tab 12: Network Vulnerability Index
with tabs[12]:
    st.header("Network Vulnerability Index (NVI)")
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

# Tab 13: 3D Network Visualization
with tabs[13]:
    st.header("3D Network Visualization (Plotly)")
    if st.session_state["string_edges"].empty:
        st.info("Build PPI network first.")
    else:
        G = build_graph_from_edges(st.session_state["string_edges"])
        fig3d = novel_feature_3_3D_viz(G)
        st.plotly_chart(fig3d, use_container_width=True)

# Tab 14: Downloads
with tabs[14]:
    st.header("Downloads & Export")
    if st.session_state["string_edges"].empty:
        st.info("No network to export.")
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
        st.download_button("Download network metrics CSV", df_metrics.to_csv(index=False).encode("utf-8"), "network_metrics.csv")
        st.download_button("Download STRING edges CSV", st.session_state["string_edges"].to_csv(index=False).encode("utf-8"), "string_edges.csv")

st.markdown("---")
st.markdown("<small>Note: external APIs (AlphaFold, UniProt, STRING) are used. Respect their rate limits.</small>", unsafe_allow_html=True)
