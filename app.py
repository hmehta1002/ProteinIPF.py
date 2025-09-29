# app.py
# Protein Network Explorer — Full + Novel Metrics + Mock Intrinsic Disorder + Heuristics + Insights
# Requires: streamlit, pandas, numpy, matplotlib, requests, plotly, networkx
# Optional: py3Dmol, biopython (SeqIO)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, json, math, random, itertools
from collections import deque
import requests
import networkx as nx
import plotly.graph_objects as go

# Optional libraries
try:
    import py3Dmol
except Exception:
    py3Dmol = None
try:
    from Bio import SeqIO
except Exception:
    SeqIO = None

# -------------------------
# Page config & header
# -------------------------
st.set_page_config(page_title="Protein Network Explorer", layout="wide", page_icon="🧬")
st.markdown(
    "<h1 style='text-align:center; color:#00CED1;'>🧬 Protein Network Explorer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#20B2AA;'>Network analysis, sequence heuristics, mock intrinsic disorder maps, 3D visualization, and novel metrics for research.</p>",
    unsafe_allow_html=True
)

# -------------------------
# Inline sample data
# -------------------------
SAMPLE_CSV = """Protein1,Protein2,TaxID
P53_HUMAN,MDM2_HUMAN,9606
P53_HUMAN,BRCA1_HUMAN,9606
BRCA1_HUMAN,BRCA2_HUMAN,9606
BRCA2_HUMAN,RAD51_HUMAN,9606
MDM2_HUMAN,UBC_HUMAN,9606
AKT1_HUMAN,MTOR_HUMAN,9606
AKT1_HUMAN,PIK3CA_HUMAN,9606
PIK3CA_HUMAN,PTEN_HUMAN,9606
PTEN_HUMAN,TP53BP1_HUMAN,9606
"""

SAMPLE_FASTA = """>P53_HUMAN
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP
>MDM2_HUMAN
MDM2SEQEXAMPLEXXXXXXXXXXXXXX
>BRCA1_HUMAN
BRCA1SEQEXAMPLEYYYYYYYYYYYYYYY
"""

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Upload / Sample Controls")
uploaded_csv = st.sidebar.file_uploader("Upload network CSV (edges)", type=["csv"])
uploaded_fasta = st.sidebar.file_uploader("Upload FASTA (optional)", type=["fasta","fa","txt"])
st.sidebar.markdown("---")
st.sidebar.download_button("Download sample CSV", SAMPLE_CSV.encode("utf-8"), "sample_network.csv")
st.sidebar.download_button("Download sample FASTA", SAMPLE_FASTA.encode("utf-8"), "sample_sequences.fasta")
if st.sidebar.button("Load sample into session"):
    st.session_state["use_sample"] = True
    st.session_state["sample_csv_text"] = SAMPLE_CSV
    st.session_state["sample_fasta_text"] = SAMPLE_FASTA
    st.sidebar.success("Sample loaded.")

filter_human = st.sidebar.checkbox("Filter Homo sapiens (TaxID 9606)", value=True)

# Column mapping
with st.sidebar.expander("Column mapping"):
    map_col_a = st.text_input("Protein A column", value="Protein1")
    map_col_b = st.text_input("Protein B column", value="Protein2")
    map_tax = st.text_input("TaxID column", value="TaxID")

# -------------------------
# Helpers
# -------------------------
def parse_fasta_text(text):
    seqs = {}
    if not text:
        return seqs
    entries = text.strip().split(">")
    for e in entries:
        if not e.strip(): continue
        lines = e.strip().splitlines()
        header = lines[0].split()[0]
        seq = "".join(lines[1:]).replace(" ", "").replace("\r", "")
        seqs[header] = seq
    return seqs

def read_csv_like(obj):
    try:
        if hasattr(obj, "read") and not isinstance(obj, str):
            content = obj.read()
            if isinstance(content, (bytes, bytearray)):
                content = content.decode("utf-8")
            return pd.read_csv(io.StringIO(content))
        elif isinstance(obj, str):
            return pd.read_csv(io.StringIO(obj))
        else:
            return None
    except Exception:
        try:
            return pd.read_csv(obj)
        except Exception:
            return None

# Sequence heuristics
AA_MASS = {'A':71.03711,'R':156.10111,'N':114.04293,'D':115.02694,'C':103.00919,'E':129.04259,'Q':128.05858,'G':57.02146,
           'H':137.05891,'I':113.08406,'L':113.08406,'K':128.09496,'M':131.04049,'F':147.06841,'P':97.05276,'S':87.03203,
           'T':101.04768,'W':186.07931,'Y':163.06333,'V':99.06841}
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
DISORDER_AA = set(list("PESQKRNTG"))

def compute_seq_metrics(seq):
    seq = seq.upper()
    count=0; mw=0.0; gravy_sum=0.0; disorder_count=0
    for aa in seq:
        if aa in AA_MASS:
            count += 1
            mw += AA_MASS[aa]
            gravy_sum += KD.get(aa,0.0)
            if aa in DISORDER_AA:
                disorder_count += 1
    if count == 0:
        return {"length":0,"mw":0.0,"gravy":0.0,"disorder_frac":0.0,"instability":0.0}
    gravy = gravy_sum/count
    disorder_frac = disorder_count/count
    instability = max(0.0, 1.0 - min(1.0, count/1000.0)) + abs(gravy)/10.0
    return {"length":count, "mw":round(mw,2), "gravy":round(gravy,4), "disorder_frac":round(disorder_frac,4), "instability":round(instability,4)}

# -------------------------
# Graph functions (pure Python)
# -------------------------
def bfs_shortest_paths(graph, start):
    visited = {start:0}
    q = deque([start])
    while q:
        v = q.popleft()
        for nb in graph.get(v, []):
            if nb not in visited:
                visited[nb] = visited[v] + 1
                q.append(nb)
    return visited

def compute_betweenness(graph, nodes):
    bet = dict.fromkeys(nodes, 0.0)
    for s in nodes:
        S=[]; P={v:[] for v in nodes}; sigma=dict.fromkeys(nodes,0.0); dist=dict.fromkeys(nodes,-1)
        sigma[s]=1.0; dist[s]=0; Q=deque([s])
        while Q:
            v = Q.popleft(); S.append(v)
            for w in graph[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1; Q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]; P[w].append(v)
        delta = dict.fromkeys(nodes, 0.0)
        while S:
            w = S.pop()
            for v in P[w]:
                if sigma[w] != 0:
                    delta_v = (sigma[v] / sigma[w]) * (1 + delta[w])
                else:
                    delta_v = 0
                delta[v] += delta_v
            if w != s:
                bet[w] += delta[w]
    return bet

def compute_clustering(graph, nodes):
    clustering = {}
    for node in nodes:
        neigh = list(graph[node])
        k = len(neigh)
        if k < 2:
            clustering[node] = 0.0
        else:
            links = 0
            for u,v in itertools.combinations(neigh,2):
                if u in graph[v]:
                    links += 1
            clustering[node] = 2*links/(k*(k-1))
    return clustering

# DFWMIN — novel metric
def compute_dfwmin(graph, nodes, seq_metrics):
    dfw = {}
    for n in nodes:
        deg = len(graph[n])
        dis = seq_metrics.get(n, {}).get("disorder_frac",0.0)
        mw = seq_metrics.get(n, {}).get("mw",1.0)
        dfw[n] = round(deg * dis / (math.log1p(mw)),4)
    return dfw

# Composite Centrality
def compute_composite(deg, bet, clust):
    comp = {}
    for k in deg:
        comp[k] = round(deg[k]*0.4 + bet[k]*0.4 + clust[k]*0.2,4)
    return comp

# -------------------------
# Load CSV
# -------------------------
if uploaded_csv or ("use_sample" in st.session_state):
    if uploaded_csv:
        df_edges = read_csv_like(uploaded_csv)
    else:
        df_edges = read_csv_like(st.session_state.get("sample_csv_text",""))
    
    # Filter human
    if filter_human and map_tax in df_edges.columns:
        df_edges = df_edges[df_edges[map_tax]==9606]
    
    nodes = set(df_edges[map_col_a]).union(set(df_edges[map_col_b]))
    graph = {n:set() for n in nodes}
    for _,r in df_edges.iterrows():
        graph[r[map_col_a]].add(r[map_col_b])
        graph[r[map_col_b]].add(r[map_col_a])
    
    st.success(f"Network loaded: {len(nodes)} proteins, {len(df_edges)} edges.")

    # -------------------------
    # Tabs
    # -------------------------
    tabs = st.tabs(["Network Graph","Metrics & Novelty","Sequences & Motifs","3D Viewer","Disorder Map","Insights & Downloads"])
    
    # -------------------------
    # Network Graph (Plotly + NetworkX)
    # -------------------------
    with tabs[0]:
        st.markdown("### Interactive Network Graph")
        try:
            G = nx.Graph()
            for a,b in zip(df_edges[map_col_a], df_edges[map_col_b]):
                G.add_edge(a,b)
            pos = nx.spring_layout(G, seed=42)
            edge_x=[]; edge_y=[]
            for edge in G.edges():
                x0,y0 = pos[edge[0]]
                x1,y1 = pos[edge[1]]
                edge_x += [x0,x1,None]
                edge_y += [y0,y1,None]
            edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1,color='#888'), hoverinfo='none', mode='lines')
            node_x=[]; node_y=[]; text=[]
            for node in G.nodes():
                x,y = pos[node]
                node_x.append(x); node_y.append(y); text.append(node)
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers+text', text=text,
                textposition="top center", hoverinfo='text',
                marker=dict(showscale=False, color='#20B2AA', size=12, line_width=2)
            )
            fig = go.Figure(data=[edge_trace,node_trace],
                            layout=go.Layout(showlegend=False, hovermode='closest',
                                             margin=dict(b=0,l=0,r=0,t=0)))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error rendering graph: {e}")

    # -------------------------
    # Metrics & Novelty
    # -------------------------
    with tabs[1]:
        st.markdown("### Node Metrics & Novelty")
        deg = {n:len(graph[n]) for n in nodes}
        bet = compute_betweenness(graph, nodes)
        clust = compute_clustering(graph, nodes)
        dfw = compute_dfwmin(graph, nodes, {})
        comp = compute_composite(deg, bet, clust)
        metrics_df = pd.DataFrame({
            "Degree": deg,
            "Betweenness": bet,
            "Clustering": clust,
            "DFWMIN": dfw,
            "CompositeCentrality": comp
        }).sort_values("CompositeCentrality", ascending=False)
        st.dataframe(metrics_df.style.background_gradient(cmap="viridis"))
    
    # -------------------------
    # Sequences & Motifs
    # -------------------------
    with tabs[2]:
        st.markdown("### Protein Sequence Heuristics")
        sequences = {}
        if uploaded_fasta:
            fasta_text = uploaded_fasta.read().decode("utf-8")
            sequences = parse_fasta_text(fasta_text)
        elif "use_sample" in st.session_state:
            sequences = parse_fasta_text(st.session_state.get("sample_fasta_text",""))
        seq_metrics = {k:compute_seq_metrics(v) for k,v in sequences.items()}
        if seq_metrics:
            st.dataframe(pd.DataFrame(seq_metrics).T.style.background_gradient(cmap="plasma"))
        else:
            st.info("Upload a FASTA to see sequence metrics.")
    
    # -------------------------
    # 3D Viewer (safe py3Dmol)
    # -------------------------
    with tabs[3]:
        st.markdown("### AlphaFold / 3D Structure Viewer (mock)")
        if py3Dmol:
            viewer = py3Dmol.view(width=600,height=500)
            viewer.addModel("ATOM      1  N   MET A   1      11.104  13.207   2.568  1.00  0.00           N","pdb")
            viewer.setStyle({"cartoon":{"color":"spectrum"}})
            viewer.zoomTo()
            viewer_html = viewer._make_html()
            st.components.v1.html(viewer_html, height=500)
        else:
            st.warning("py3Dmol not installed — 3D viewer not available.")
    
    # -------------------------
    # Disorder Map
    # -------------------------
    with tabs[4]:
        st.markdown("### Mock Intrinsic Disorder Map")
        if seq_metrics:
            fig, ax = plt.subplots(figsize=(12,4))
            for k,v in seq_metrics.items():
                dis_profile = [random.random()*v["disorder_frac"] for _ in range(v["length"])]
                ax.plot(range(1,v["length"]+1), dis_profile, label=k)
            ax.set_xlabel("Residue")
            ax.set_ylabel("Disorder Propensity")
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Sequence metrics not available.")
    
    # -------------------------
    # Insights & Downloads
    # -------------------------
    with tabs[5]:
        st.markdown("### Insights & Downloads")
        if 'metrics_df' in locals():
            st.markdown("**Top 5 proteins by Composite Centrality:**")
            st.table(metrics_df.head(5))
            st.download_button("Download metrics CSV", metrics_df.to_csv().encode("utf-8"), "metrics.csv")
            st.download_button("Download metrics JSON", metrics_df.to_json().encode("utf-8"), "metrics.json")
        st.markdown("You can explore multiple disease datasets, apply filters, and inspect community influence metrics.")
else:
    st.info("Upload a CSV file to start network analysis or load the sample CSV from sidebar.")
