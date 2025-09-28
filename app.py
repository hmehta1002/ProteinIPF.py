# app.py
# Protein Network Explorer — Full + Mock Intrinsic Disorder + Heuristics
# Required: streamlit, pandas, matplotlib, numpy, requests
# Optional: py3Dmol, pyvis, biopython (SeqIO) — app will run without them.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, json, math, random, itertools, time
from collections import deque, defaultdict
import requests
import streamlit.components.v1 as components

# Optional libraries (used only if installed)
try:
    import py3Dmol
except Exception:
    py3Dmol = None
try:
    from pyvis.network import Network
except Exception:
    Network = None
try:
    from Bio import SeqIO
except Exception:
    SeqIO = None

# -------------------------
# Page config & header
# -------------------------
st.set_page_config(page_title="Protein Network Explorer", layout="wide")
st.markdown("<h1 style='text-align:center; color:#4B0082;'>🧬 Protein Network Explorer </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6A5ACD;'>All features included — mock intrinsic-disorder map + sequence heuristics retained.</p>", unsafe_allow_html=True)

# -------------------------
# Inline sample data (always available)
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
# Sidebar: uploads & sample controls
# -------------------------
st.sidebar.header("Upload / Sample Controls")
uploaded_csv = st.sidebar.file_uploader("Upload network CSV (edges)", type=["csv"], key="uploader_edges")
uploaded_fasta = st.sidebar.file_uploader("Upload FASTA (optional)", type=["fasta","fa","txt"], key="uploader_fasta")
st.sidebar.markdown("---")
st.sidebar.download_button("Download sample CSV", SAMPLE_CSV.encode("utf-8"), "sample_network.csv", key="dl_sample_csv")
st.sidebar.download_button("Download sample FASTA", SAMPLE_FASTA.encode("utf-8"), "sample_sequences.fasta", key="dl_sample_fasta")

if st.sidebar.button("Load sample into session", key="load_sample_btn"):
    st.session_state["use_sample"] = True
    st.session_state["sample_csv_text"] = SAMPLE_CSV
    st.session_state["sample_fasta_text"] = SAMPLE_FASTA
    st.sidebar.success("Sample loaded to session memory.", key="sample_loaded_success")

st.sidebar.markdown("---")
filter_human = st.sidebar.checkbox("Filter to Homo sapiens (TaxID 9606)", value=True, key="filter_human")
with st.sidebar.expander("Column mapping (if headers differ)", expanded=False):
    map_col_a = st.text_input("Protein A column", value="Protein1", key="map_col_a")
    map_col_b = st.text_input("Protein B column", value="Protein2", key="map_col_b")
    map_tax = st.text_input("TaxID column", value="TaxID", key="map_tax")

# -------------------------
# Helpers: FASTA parse, CSV read, seq heuristics
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

# Sequence heuristic metrics (GRAVY, disorder fraction, instability proxy)
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
# Pure-Python graph functions (no networkx required)
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
            clustering[node] = 2 * links / (k * (k - 1))
    return clustering

def label_propagation(graph, max_iter=200):
    labels = {n:n for n in graph}
    nodes = list(graph.keys())
    for it in range(max_iter):
        changed = False
        random.shuffle(nodes)
        for node in nodes:
            counts = defaultdict(int)
            for nb in graph[node]:
                counts[labels[nb]] += 1
            if not counts:
                continue
            maxc = max(counts.values())
            best = [lab for lab,c in counts.items() if c==maxc]
            new_label = random.choice(best)
            if labels[node] != new_label:
                labels[node] = new_label
                changed = True
        if not changed:
            break
    unique = {}; communities = defaultdict(list); idx = 0
    for n,lbl in labels.items():
        if lbl not in unique:
            unique[lbl] = idx; idx += 1
        communities[unique[lbl]].append(n)
    return labels, dict(communities)

def spring_layout(graph, iterations=150, width=1.0, height=1.0):
    nodes = list(graph.keys()); N=len(nodes)
    k = math.sqrt((width*height)/max(1,N))
    pos = {n:[random.uniform(0,width), random.uniform(0,height)] for n in nodes}
    t = width/10.0; dt = t/(iterations+1)
    def rep(d): return (k*k)/d if d!=0 else k*k
    def attr(d): return (d*d)/k
    for i in range(iterations):
        disp = {n:[0.0,0.0] for n in nodes}
        for a in nodes:
            for b in nodes:
                if a==b: continue
                dx = pos[a][0]-pos[b][0]; dy = pos[a][1]-pos[b][1]; dist = math.hypot(dx,dy) + 1e-9
                f = rep(dist)
                disp[a][0] += (dx/dist)*f; disp[a][1] += (dy/dist)*f
        for a in nodes:
            for b in graph[a]:
                dx = pos[a][0]-pos[b][0]; dy = pos[a][1]-pos[b][1]; dist = math.hypot(dx,dy) + 1e-9
                f = attr(dist)
                disp[a][0] -= (dx/dist)*f; disp[a][1] -= (dy/dist)*f
        for n in nodes:
            dx,dy = disp[n]; dlen = math.hypot(dx,dy)
            if dlen > 0:
                pos[n][0] += (dx/dlen)*min(dlen,t); pos[n][1] += (dy/dlen)*min(dlen,t)
            pos[n][0] = min(width, max(0.0, pos[n][0])); pos[n][1] = min(height, max(0.0, pos[n][1]))
        t -= dt
    # normalize
    xs = [pos[n][0] for n in nodes]; ys = [pos[n][1] for n in nodes]
    minx,maxx = min(xs), max(xs); miny,maxy = min(ys), max(ys)
    for n in nodes:
        pos[n][0] = (pos[n][0]-minx)/(maxx-minx+1e-12) if maxx-minx>0 else 0.5
        pos[n][1] = (pos[n][1]-miny)/(maxy-miny+1e-12) if maxy-miny>0 else 0.5
    return pos

# -------------------------
# Load CSV & FASTA (priority: session-sample -> uploaded -> inline sample)
# -------------------------
if st.session_state.get("use_sample", False) and st.session_state.get("sample_csv_text"):
    try:
        df = pd.read_csv(io.StringIO(st.session_state["sample_csv_text"]))
    except Exception:
        df = pd.read_csv(io.StringIO(SAMPLE_CSV))
elif uploaded_csv is not None:
    df = read_csv_like(uploaded_csv)
    if df is None:
        try:
            df = pd.read_csv(uploaded_csv)
        except Exception:
            df = pd.read_csv(io.StringIO(SAMPLE_CSV))
else:
    df = pd.read_csv(io.StringIO(SAMPLE_CSV))

# fasta
if st.session_state.get("use_sample", False) and st.session_state.get("sample_fasta_text"):
    fasta_seqs = parse_fasta_text(st.session_state["sample_fasta_text"])
else:
    if uploaded_fasta is not None:
        try:
            raw = uploaded_fasta.read()
            txt = raw.decode("utf-8") if isinstance(raw, (bytes,bytearray)) else str(raw)
            fasta_seqs = parse_fasta_text(txt)
        except Exception:
            fasta_seqs = parse_fasta_text(SAMPLE_FASTA)
    else:
        fasta_seqs = parse_fasta_text(SAMPLE_FASTA)

# show columns
st.sidebar.markdown("**Detected CSV columns:**")
try:
    st.sidebar.write(list(df.columns))
except Exception:
    st.sidebar.write("Could not detect columns.")

# -------------------------
# Build adjacency (mapping)
# -------------------------
cols = list(df.columns)
def choose_col(preferred, fallbacks, available):
    if preferred in available:
        return preferred
    for f in fallbacks:
        if f in available:
            return f
    return None

protA_col = choose_col(map_col_a, ["Protein1","protein1","protA","InteractorA","source","geneA","A"], cols)
protB_col = choose_col(map_col_b, ["Protein2","protein2","protB","InteractorB","target","geneB","B"], cols)
tax_col = choose_col(map_tax, ["TaxID","taxid","tax_id","Tax_Id"], cols)

adj = {}
proteins = set()
parse_errors = []
if protA_col is None or protB_col is None:
    parse_errors.append(f"Could not find protein columns. Detected: {cols}. Use mapping in sidebar.")
else:
    for _, row in df.iterrows():
        try:
            p1 = str(row.get(protA_col, "")).strip(); p2 = str(row.get(protB_col, "")).strip()
            if filter_human and tax_col:
                taxv = str(row.get(tax_col, "")).strip()
                if taxv and taxv not in ["9606","9606.0","9606.00"]:
                    continue
            if p1 and p2 and p1.lower() not in ["nan","none"] and p2.lower() not in ["nan","none"]:
                proteins.update([p1,p2])
                adj.setdefault(p1, set()).add(p2)
                adj.setdefault(p2, set()).add(p1)
        except Exception:
            continue

# fallback sample adjacency if empty
if not adj:
    df_sample = pd.read_csv(io.StringIO(SAMPLE_CSV))
    adj = {}; proteins = set()
    for _, r in df_sample.iterrows():
        p1 = r["Protein1"]; p2 = r["Protein2"]
        proteins.update([p1,p2])
        adj.setdefault(p1, set()).add(p2)
        adj.setdefault(p2, set()).add(p1)
    if not fasta_seqs:
        fasta_seqs = parse_fasta_text(SAMPLE_FASTA)

# -------------------------
# Compute metrics (pure Python)
# -------------------------
degree = {n: len(adj[n]) for n in proteins}
closeness = {}
for n in proteins:
    sp = bfs_shortest_paths(adj, n)
    if len(sp) > 1:
        total = sum(sp.values())
        closeness[n] = (len(sp)-1)/total if total > 0 else 0.0
    else:
        closeness[n] = 0.0

betweenness = compute_betweenness(adj, list(proteins))
clustering = compute_clustering(adj, list(proteins))

metrics_df = pd.DataFrame({
    "Protein": list(proteins),
    "Degree": [degree[p] for p in proteins],
    "Closeness": [closeness[p] for p in proteins],
    "Betweenness": [betweenness[p] for p in proteins],
    "Clustering": [clustering[p] for p in proteins]
}).sort_values(by="Degree", ascending=False).reset_index(drop=True)

# -------------------------
# Communities & layout
# -------------------------
labels, communities = label_propagation(adj, max_iter=150)
num_communities = len(communities)
positions = spring_layout(adj, iterations=120, width=1.0, height=1.0)

# -------------------------
# UI: Tabs (strings only)
# -------------------------
tab_names = [
    "Upload / Files",
    "Network Map (Clusters)",
    "Network Metrics",
    "Protein Details",
    "Sequences",
    "Motifs / Domains",
    "3D Viewer",
    "Intrinsic Disorder & Stability",
    "Downloads"
]
tabs = st.tabs(tab_names)

# -------------------------
# Tab: Upload / Files
# -------------------------
with tabs[0]:
    st.header("Upload / Files")
    st.markdown("Upload CSV (edges) and optional FASTA. Use sidebar to load sample data.")
    st.write(f"Detected columns: {cols}")
    st.write(f"Nodes: {len(proteins)} | Edges (approx): {sum(len(v) for v in adj.values())//2} | Communities: {num_communities}")
    if parse_errors:
        for msg in parse_errors:
            st.error(msg)
    if st.button("Reset to sample", key="reset_main"):
        st.session_state["use_sample"] = True
        st.session_state["sample_csv_text"] = SAMPLE_CSV
        st.session_state["sample_fasta_text"] = SAMPLE_FASTA
        st.experimental_rerun()

# -------------------------
# Tab: Network Map (Clusters)
# -------------------------
with tabs[1]:
    st.header("Network Map (Clusters)")
    fig, ax = plt.subplots(figsize=(12,8))
    cmap = plt.cm.get_cmap("tab20", max(4, num_communities))
    drawn = set()
    for a in adj:
        for b in adj[a]:
            if (a,b) in drawn or (b,a) in drawn: continue
            xa,ya = positions[a]; xb,yb = positions[b]
            ax.plot([xa,xb],[ya,yb], color="#DDDDDD", linewidth=0.8, zorder=1, alpha=0.6)
            drawn.add((a,b))
    for node in sorted(list(proteins)):
        x,y = positions[node]
        comm_idx = None
        for cid,members in communities.items():
            if node in members:
                comm_idx = cid; break
        color = cmap(comm_idx % cmap.N) if comm_idx is not None else (0.6,0.6,0.6)
        sz = 50 + degree[node]*20
        ax.scatter(x,y,s=sz,color=color,edgecolor='black',linewidth=0.6,zorder=3)
        ax.text(x,y+0.02,node,fontsize=8,ha='center',rotation=30,zorder=4)
    ax.axis('off')
    st.pyplot(fig)

    st.markdown("Select node to inspect metrics & sequence")
    sel_node = st.selectbox("Select protein", options=sorted(list(proteins)), key="map_select")
    if sel_node:
        st.subheader(sel_node)
        st.write(f"- Degree: **{degree[sel_node]}**")
        st.write(f"- Closeness: **{closeness[sel_node]:.6f}**")
        st.write(f"- Betweenness: **{betweenness[sel_node]:.6f}**")
        st.write(f"- Clustering: **{clustering[sel_node]:.6f}**")
        seq = fasta_seqs.get(sel_node, "")
        if seq:
            st.text_area("Sequence (FASTA)", seq, height=200, key="map_seq_area")
        else:
            with st.spinner("Attempting UniProt fetch..."):
                try:
                    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{sel_node}.fasta", timeout=10)
                    if r.ok and r.text.strip():
                        seq = "".join(r.text.splitlines()[1:])
                        st.text_area("Sequence (UniProt)", seq, height=200, key="map_seq_uniprot")
                    else:
                        st.warning("Sequence not found (UniProt return empty).")
                except Exception:
                    st.warning("UniProt lookup failed (network/rate).")
        if seq:
            seq_metrics = compute_seq_metrics(seq)
            st.markdown("**Sequence heuristics**")
            st.write(f"- Length: **{seq_metrics['length']}**")
            st.write(f"- GRAVY: **{seq_metrics['gravy']}**")
            st.write(f"- Disorder fraction: **{seq_metrics['disorder_frac']}**")
            st.write(f"- Instability proxy: **{seq_metrics['instability']}**")

# -------------------------
# Tab: Network Metrics
# -------------------------
with tabs[2]:
    st.header("Network Metrics")
    st.dataframe(metrics_df, height=420)
    st.download_button("Download metrics CSV", metrics_df.to_csv(index=False).encode("utf-8"), "protein_metrics.csv", key="dl_metrics")

# -------------------------
# Tab: Protein Details
# -------------------------
with tabs[3]:
    st.header("Protein Details")
    pid = st.text_input("Enter UniProt entry name / ID (e.g., P53_HUMAN)", key="prot_lookup")
    pick = st.selectbox("Or pick from network", options=[""] + sorted(list(proteins)), key="prot_pick")
    target = pid.strip() if pid.strip() else (pick if pick else "")
    if target:
        st.subheader(target)
        if target in degree:
            st.write(f"- Degree: **{degree[target]}**")
            st.write(f"- Closeness: **{closeness[target]:.6f}**")
            st.write(f"- Betweenness: **{betweenness[target]:.6f}**")
            st.write(f"- Clustering: **{clustering[target]:.6f}**")
        else:
            st.info("Protein not in current network.")
        seq_txt = fasta_seqs.get(target, "")
        if seq_txt:
            st.text_area("Sequence", seq_txt, height=250, key="details_seq")
        else:
            with st.spinner("Fetching UniProt FASTA..."):
                try:
                    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{target}.fasta", timeout=12)
                    if r.ok and r.text.strip():
                        seq_txt = "".join(r.text.splitlines()[1:])
                        st.text_area("Sequence (UniProt)", seq_txt, height=250, key="details_seq_uniprot")
                    else:
                        st.warning("No sequence on UniProt.")
                except Exception:
                    st.warning("UniProt fetch failed.")
        if st.button("Export this protein", key="export_prot"):
            out = {"Protein": target, "Degree": degree.get(target,0), "Closeness": closeness.get(target,0.0), "Betweenness": betweenness.get(target,0.0), "Clustering": clustering.get(target,0.0), "Sequence": seq_txt}
            st.download_button("Download protein JSON", json.dumps(out, indent=2).encode("utf-8"), f"{target}_data.json", key="dl_prot_json")

# -------------------------
# Tab: Sequences
# -------------------------
with tabs[4]:
    st.header("Sequences / FASTA")
    fasta_up = st.file_uploader("Upload FASTA (optional)", type=["fasta","fa","txt"], key="fasta_up")
    if fasta_up:
        try:
            if SeqIO is not None:
                recs = list(SeqIO.parse(fasta_up, "fasta"))
                for rec in recs:
                    st.text(f">{rec.id}\n{rec.seq}")
            else:
                raw = fasta_up.read()
                txt = raw.decode("utf-8") if isinstance(raw, (bytes,bytearray)) else str(raw)
                st.text(txt)
        except Exception as e:
            st.error(f"Failed to parse FASTA: {e}")
    else:
        st.info("No FASTA uploaded — sample shown")
        st.text(SAMPLE_FASTA)

# -------------------------
# Tab: Motifs / Domains (UniProt REST)
# -------------------------
with tabs[5]:
    st.header("Motifs & Domains (UniProt)")
    motif_ids = st.text_input("Enter UniProt ID(s), comma-separated (e.g., P69905)", key="motif_ids")
    if st.button("Fetch features", key="fetch_features"):
        ids = [i.strip() for i in motif_ids.split(",") if i.strip()]
        if not ids:
            st.warning("Enter at least one UniProt ID.")
        for uid in ids:
            with st.spinner(f"Fetching {uid}..."):
                url = f"https://rest.uniprot.org/uniprotkb/{uid}.json"
                try:
                    r = requests.get(url, timeout=20)
                    if r.ok:
                        data = r.json()
                        feats = data.get("features", [])
                        if not feats:
                            st.info(f"No features for {uid}.")
                        else:
                            st.success(f"Features for {uid}: {len(feats)}")
                            for f in feats[:100]:
                                ftype = f.get("type","")
                                desc = f.get("description","") or f.get("comment","")
                                loc = f.get("location", {})
                                start = loc.get("start", loc.get("begin",""))
                                end = loc.get("end", loc.get("end",""))
                                st.markdown(f"**{ftype}** — {desc}  \nLocation: {start} — {end}")
                    else:
                        st.error(f"Failed to fetch {uid} (status {r.status_code})")
                except Exception as e:
                    st.error(f"Failed to fetch {uid}: {e}")

# -------------------------
# Tab: 3D Viewer (AlphaFold)
# -------------------------
with tabs[6]:
    st.header("3D Viewer (AlphaFold)")
    af_id = st.text_input("Enter AlphaFold/UniProt entry (e.g., P69905)", key="af_input")
    if st.button("Load AlphaFold", key="load_af"):
        if not af_id.strip():
            st.warning("Enter an entry.")
        else:
            if py3Dmol is None:
                st.error("py3Dmol not installed. Use AlphaFold website instead.")
                st.markdown(f"https://alphafold.ebi.ac.uk/entry/{af_id}")
            else:
                with st.spinner("Fetching AlphaFold model..."):
                    try:
                        url = f"https://alphafold.ebi.ac.uk/files/AF-{af_id}-F1-model_v4.pdb"
                        r = requests.get(url, timeout=15)
                        if r.ok and r.text:
                            pdb = r.text
                            view = py3Dmol.view(width=800, height=500)
                            view.addModel(pdb, "pdb")
                            view.setStyle({'cartoon': {'color':'spectrum'}})
                            view.zoomTo()
                            view.show()
                            try:
                                html = view._make_html()
                                components.html(html, height=520)
                            except Exception:
                                components.html(view.js(), height=520)
                        else:
                            st.error("AlphaFold model not found for that ID.")
                    except Exception as e:
                        st.error(f"AlphaFold fetch failed: {e}")

# -------------------------
# Tab: Intrinsic Disorder & Stability (MOCK + heuristics)
# -------------------------
with tabs[7]:
    st.header("Intrinsic Disorder Map (mock) + Sequence Heuristics")
    st.markdown("This uses **mock disorder scores** (random) for the curve, and **sequence heuristics** (GRAVY, disorder fraction, instability proxy) are computed from sequence.")

    dis_id = st.text_input("Enter UniProt entry or protein name for disorder map (or pick from network)", key="disorder_id")
    pick_dis = st.selectbox("Or pick from network", options=[""] + sorted(list(proteins)), key="pick_dis")
    id_use = dis_id.strip() if dis_id.strip() else (pick_dis if pick_dis else "")

    if st.button("Show Intrinsic Disorder Map (mock)", key="show_dis_map"):
        seq_text = ""
        if id_use and id_use in fasta_seqs:
            seq_text = fasta_seqs[id_use]
        elif id_use and id_use.upper().startswith(">"):
            parsed = parse_fasta_text(id_use)
            if parsed:
                seq_text = list(parsed.values())[0]
        elif id_use:
            try:
                r = requests.get(f"https://rest.uniprot.org/uniprotkb/{id_use}.fasta", timeout=10)
                if r.ok and r.text.strip():
                    seq_text = "".join(r.text.splitlines()[1:])
            except Exception:
                seq_text = ""
        # If still no sequence, try first protein in list as fallback
        if not seq_text and proteins:
            fallback = sorted(list(proteins))[0]
            seq_text = fasta_seqs.get(fallback, "")
            st.info(f"No sequence found for input. Using fallback sequence: {fallback}")

        if not seq_text:
            st.error("No sequence available to build disorder map. Upload FASTA or ensure UniProt ID / name is valid.")
        else:
            # Mock disorder scores: random but smoothed to look realistic
            L = len(seq_text)
            rng = np.random.RandomState(seed=sum(ord(c) for c in seq_text) % 2**32)
            raw = rng.rand(L)
            # smooth with moving average
            window = max(3, min(25, L//20))
            kernel = np.ones(window)/window
            smooth = np.convolve(raw, kernel, mode='same')
            # normalize to 0..1
            smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min() + 1e-12)

            # Plot disorder map
            fig, ax = plt.subplots(figsize=(10,3))
            ax.plot(range(1, L+1), smooth, linewidth=1.5)
            ax.axhline(0.5, color='red', linestyle='--', label='Disorder threshold (0.5)')
            ax.set_xlabel('Residue position')
            ax.set_ylabel('Mock disorder score')
            ax.set_title(f'Intrinsic Disorder Map (mock) — sequence length {L}')
            ax.set_ylim(0,1)
            ax.legend()
            st.pyplot(fig)

            # Also compute and show heuristics
            heur = compute_seq_metrics(seq_text)
            st.markdown("**Sequence heuristics**")
            st.write(f"- Length: **{heur['length']}**")
            st.write(f"- Approx. MW: **{heur['mw']} Da**")
            st.write(f"- GRAVY: **{heur['gravy']}**")
            st.write(f"- Disorder-promoting fraction (heuristic): **{heur['disorder_frac']}**")
            st.write(f"- Instability proxy (heuristic): **{heur['instability']}**")
            st.info("Note: Disorder curve is mock (offline). Heuristics are sequence-derived estimates; for publication-grade use IUPred/ProtParam etc.")

# -------------------------
# Tab: Downloads
# -------------------------
with tabs[8]:
    st.header("Downloads & Exports")
    st.download_button("Download sample CSV", SAMPLE_CSV.encode("utf-8"), "sample_network.csv", key="dl_tab_sample_csv")
    st.download_button("Download sample FASTA", SAMPLE_FASTA.encode("utf-8"), "sample_sequences.fasta", key="dl_tab_sample_fasta")
    st.download_button("Download metrics CSV", metrics_df.to_csv(index=False).encode("utf-8"), "protein_metrics.csv", key="dl_tab_metrics")
    sel = st.selectbox("Select protein to export", options=[""] + sorted(list(proteins)), key="export_sel")
    if st.button("Export selected protein", key="export_sel_btn"):
        if sel:
            seq = fasta_seqs.get(sel, "")
            if not seq:
                try:
                    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{sel}.fasta", timeout=6)
                    if r.ok and r.text.strip():
                        seq = "".join(r.text.splitlines()[1:])
                except Exception:
                    seq = ""
            out = {"Protein": sel, "Degree": degree.get(sel,0), "Closeness": closeness.get(sel,0.0), "Betweenness": betweenness.get(sel,0.0), "Clustering": clustering.get(sel,0.0), "Sequence": seq}
            st.download_button("Download JSON", json.dumps(out, indent=2).encode("utf-8"), f"{sel}_data.json", key="dl_export_sel")
        else:
            st.warning("Choose a protein first.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>This single-file app includes all requested features and a mock intrinsic disorder map (offline). If anything errors, paste the full traceback here and I will fix it immediately.</p>", unsafe_allow_html=True)
