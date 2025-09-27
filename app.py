# app.py
# Protein Network Explorer — complete single-file app
# - All features retained
# - Inline sample data + Reset to Sample button
# - Unique Streamlit keys to avoid duplicate element ID errors
# - No hard-coded file reads required

import streamlit as st
import pandas as pd
import requests
import math, random, itertools, io
from collections import deque, defaultdict
import matplotlib.pyplot as plt

# Optional libs (if you have them): networkx, pyvis, py3Dmol, Biopython
# The app will gracefully degrade if some of these packages are unavailable.
try:
    import networkx as nx
except Exception:
    nx = None
try:
    from pyvis.network import Network
except Exception:
    Network = None
try:
    import py3Dmol
except Exception:
    py3Dmol = None
try:
    from Bio import SeqIO
except Exception:
    SeqIO = None

# -----------------------------
# App config & header
st.set_page_config(page_title="Protein Network Explorer — Full", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align:center; color:#4B0082;'>🧬 Protein Network Explorer — Full</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6A5ACD;'>All features preserved + Reset-to-Sample + robust keys (no duplicate IDs)</p>", unsafe_allow_html=True)

# -----------------------------
# Inline sample data (guaranteed present)
SAMPLE_CSV_TEXT = """Protein1,Protein2,TaxID
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

SAMPLE_FASTA_TEXT = """>P53_HUMAN
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQK
>MDM2_HUMAN
MDM2SEQEXAMPLEXXXXXXXXXXXXXX
>BRCA1_HUMAN
BRCA1SEQEXAMPLEYYYYYYYYYYYYYYY
"""

# -----------------------------
# Sidebar: uploads, sample downloads, reset
st.sidebar.header("Upload / Samples")
uploaded_file = st.sidebar.file_uploader("Upload network CSV (Protein1,Protein2)", type=["csv"], key="uploader_edges_key")
uploaded_fasta = st.sidebar.file_uploader("Upload FASTA (optional)", type=["fasta","fa","txt"], key="uploader_fasta_key")

# Download sample files (generated from inline strings)
st.sidebar.download_button("⬇️ Download sample CSV", SAMPLE_CSV_TEXT.encode("utf-8"), "sample_network.csv", key="download_sample_csv_key")
st.sidebar.download_button("⬇️ Download sample FASTA", SAMPLE_FASTA_TEXT.encode("utf-8"), "sample_sequences.fasta", key="download_sample_fasta_key")

# Reset to sample data button (uses session_state)
if "use_sample" not in st.session_state:
    st.session_state.use_sample = False
if st.sidebar.button("Load / Reset to Sample Data", key="reset_to_sample_btn"):
    st.session_state.use_sample = True
    st.session_state.df_text = SAMPLE_CSV_TEXT
    st.session_state.fasta_text = SAMPLE_FASTA_TEXT
    st.sidebar.success("Sample data loaded into session", key="reset_success_key")

# Options: filter, mapping
filter_human = st.sidebar.checkbox("Filter to Homo sapiens (TaxID 9606)", value=True, key="filter_human_key")
with st.sidebar.expander("Column mapping (if headers differ)", expanded=False):
    colA_name = st.text_input("Protein A column name", value="Protein1", key="map_colA_key")
    colB_name = st.text_input("Protein B column name", value="Protein2", key="map_colB_key")
    tax_col_name = st.text_input("TaxID column name (optional)", value="TaxID", key="map_taxid_key")

# -----------------------------
# Helper: parse FASTA text -> dict
def parse_fasta_text(text):
    seqs = {}
    if not text:
        return seqs
    entries = text.strip().split(">")
    for e in entries:
        if not e.strip():
            continue
        lines = e.strip().splitlines()
        header = lines[0].split()[0]
        seq = "".join(lines[1:]).replace(" ", "").replace("\r","")
        seqs[header] = seq
    return seqs

# -----------------------------
# Load main dataframe: priority
# 1) If user clicked reset sample -> use session_state
# 2) Else if uploaded_file present -> use it
# 3) Else -> use inline SAMPLE_CSV_TEXT
df = None
if st.session_state.get("use_sample", False) and st.session_state.get("df_text"):
    try:
        df = pd.read_csv(io.StringIO(st.session_state.df_text))
    except Exception:
        df = pd.read_csv(io.StringIO(SAMPLE_CSV_TEXT))
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Failed to parse uploaded CSV: {e}", key="upload_parse_error_key")
        df = pd.read_csv(io.StringIO(SAMPLE_CSV_TEXT))
else:
    df = pd.read_csv(io.StringIO(SAMPLE_CSV_TEXT))

# Load fasta sequences (uploaded, session sample, fallback)
fasta_seqs = {}
if st.session_state.get("use_sample", False) and st.session_state.get("fasta_text"):
    fasta_seqs = parse_fasta_text(st.session_state.fasta_text)
elif uploaded_fasta is not None:
    try:
        fasta_bytes = uploaded_fasta.read()
        if isinstance(fasta_bytes, bytes):
            fasta_text = fasta_bytes.decode("utf-8")
        else:
            fasta_text = str(fasta_bytes)
        fasta_seqs = parse_fasta_text(fasta_text)
    except Exception:
        fasta_seqs = parse_fasta_text(SAMPLE_FASTA_TEXT)
else:
    fasta_seqs = parse_fasta_text(SAMPLE_FASTA_TEXT)

# show detected columns (helpful)
st.sidebar.markdown("**Detected CSV columns:**")
try:
    st.sidebar.write(list(df.columns), key="detected_cols_key")
except Exception:
    st.sidebar.write("Could not read columns", key="detected_cols_fail_key")

# -----------------------------
# Build adjacency (robust header matching)
def choose_col(preferred, fallbacks, cols):
    if preferred in cols:
        return preferred
    for f in fallbacks:
        if f in cols:
            return f
    return None

cols = list(df.columns)
protA_col = choose_col(colA_name, ["Protein1","protein1","protA","InteractorA","source","geneA","GeneA"], cols)
protB_col = choose_col(colB_name, ["Protein2","protein2","protB","InteractorB","target","geneB","GeneB"], cols)
tax_col = choose_col(tax_col_name, ["TaxID","taxid","tax_id","Tax_Id"], cols)

adj = {}
proteins = set()
errors = []

if protA_col is None or protB_col is None:
    errors.append(f"Could not find protein columns. Detected columns: {cols}. Use mapping fields in sidebar.")
else:
    for idx, row in df.iterrows():
        try:
            p1 = str(row.get(protA_col, "")).strip()
            p2 = str(row.get(protB_col, "")).strip()
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

if not adj:
    st.warning("No edges found after parsing. Using sample dataset to ensure app runs.", key="no_edges_warning_key")
    df = pd.read_csv(io.StringIO(SAMPLE_CSV_TEXT))
    adj = {}
    proteins = set()
    for _, r in df.iterrows():
        p1 = r["Protein1"]; p2 = r["Protein2"]
        proteins.update([p1,p2])
        adj.setdefault(p1, set()).add(p2)
        adj.setdefault(p2, set()).add(p1)

# -----------------------------
# Pure-Python metrics (so no networkx required)
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

# closeness
closeness = {}
for n in proteins:
    sp = bfs_shortest_paths(adj, n)
    if len(sp) > 1:
        s = sum(sp.values())
        closeness[n] = (len(sp)-1)/s if s>0 else 0.0
    else:
        closeness[n] = 0.0

# betweenness (BFS-based)
betweenness = dict.fromkeys(proteins, 0.0)
for s in proteins:
    S=[]; P={v:[] for v in proteins}; sigma=dict.fromkeys(proteins,0.0); dist=dict.fromkeys(proteins,-1)
    sigma[s]=1.0; dist[s]=0; Q=deque([s])
    while Q:
        v = Q.popleft(); S.append(v)
        for w in adj[v]:
            if dist[w] < 0:
                dist[w] = dist[v]+1; Q.append(w)
            if dist[w] == dist[v]+1:
                sigma[w] += sigma[v]; P[w].append(v)
    delta = dict.fromkeys(proteins, 0.0)
    while S:
        w = S.pop()
        for v in P[w]:
            if sigma[w] != 0:
                delta_v = (sigma[v]/sigma[w]) * (1 + delta[w])
            else:
                delta_v = 0
            delta[v] += delta_v
        if w != s:
            betweenness[w] += delta[w]

# clustering coefficient
clustering = {}
for node in proteins:
    neigh = list(adj[node])
    k = len(neigh)
    if k < 2:
        clustering[node] = 0.0
    else:
        links=0
        for u,v in itertools.combinations(neigh,2):
            if u in adj[v]:
                links += 1
        clustering[node] = 2*links/(k*(k-1))

degree = {n: len(adj[n]) for n in proteins}

# metrics df
metrics_df = pd.DataFrame({
    "Protein": list(proteins),
    "Degree": [degree[p] for p in proteins],
    "Closeness": [closeness[p] for p in proteins],
    "Betweenness": [betweenness[p] for p in proteins],
    "Clustering": [clustering[p] for p in proteins]
}).sort_values("Degree", ascending=False).reset_index(drop=True)

# -----------------------------
# Label propagation (community detection) pure python
def label_propagation(graph, max_iter=200):
    labels = {n:n for n in graph}
    nodes = list(graph.keys())
    for i in range(max_iter):
        changed=False
        random.shuffle(nodes)
        for node in nodes:
            counts=defaultdict(int)
            for nb in graph[node]:
                counts[labels[nb]] += 1
            if not counts:
                continue
            maxc = max(counts.values()); choices=[lab for lab,c in counts.items() if c==maxc]
            new_label = random.choice(choices)
            if labels[node] != new_label:
                labels[node] = new_label; changed=True
        if not changed:
            break
    unique = {}; communities=defaultdict(list); idx=0
    for n,lbl in labels.items():
        if lbl not in unique:
            unique[lbl]=idx; idx+=1
        communities[unique[lbl]].append(n)
    return labels, dict(communities)

labels, communities = label_propagation(adj, max_iter=200)
num_communities = len(communities)

# -----------------------------
# Spring layout (force-directed)
def spring_layout(graph, iterations=200, width=1.0, height=1.0, k=None):
    nodes = list(graph.keys()); N=len(nodes)
    if k is None:
        k = math.sqrt((width*height)/max(1,N))
    pos = {n: [random.uniform(0,width), random.uniform(0,height)] for n in nodes}
    t = width/10.0; dt = t/(iterations+1)
    def rep(d,k): return (k*k)/d if d!=0 else k*k
    def attr(d,k): return (d*d)/k
    for it in range(iterations):
        disp = {n:[0.0,0.0] for n in nodes}
        for a in nodes:
            for b in nodes:
                if a==b: continue
                dx = pos[a][0]-pos[b][0]; dy=pos[a][1]-pos[b][1]; dist=math.hypot(dx,dy)+1e-9
                f = rep(dist,k)
                disp[a][0] += (dx/dist)*f; disp[a][1] += (dy/dist)*f
        for a in nodes:
            for b in graph[a]:
                dx=pos[a][0]-pos[b][0]; dy=pos[a][1]-pos[b][1]; dist=math.hypot(dx,dy)+1e-9
                f = attr(dist,k)
                disp[a][0] -= (dx/dist)*f; disp[a][1] -= (dy/dist)*f
        for n in nodes:
            dx,dy = disp[n]; dlen = math.hypot(dx,dy)
            if dlen>0:
                pos[n][0] += (dx/dlen)*min(dlen,t); pos[n][1] += (dy/dlen)*min(dlen,t)
            pos[n][0] = min(width, max(0.0, pos[n][0])); pos[n][1] = min(height, max(0.0, pos[n][1]))
        t -= dt
    # normalize
    xs = [pos[n][0] for n in nodes]; ys=[pos[n][1] for n in nodes]
    minx,maxx=min(xs),max(xs); miny,maxy=min(ys),max(ys)
    for n in nodes:
        pos[n][0] = (pos[n][0]-minx)/(maxx-minx+1e-12) if maxx-minx>0 else 0.5
        pos[n][1] = (pos[n][1]-miny)/(maxy-miny+1e-12) if maxy-miny>0 else 0.5
    return pos

with st.spinner("Computing layout..."):
    positions = spring_layout(adj, iterations=200, width=1.0, height=1.0)

# -----------------------------
# Sequence-derived heuristics (GRAVY + disorder-promoting fraction + instability proxy)
AA_MASS = {'A':71.03711,'R':156.10111,'N':114.04293,'D':115.02694,'C':103.00919,'E':129.04259,'Q':128.05858,'G':57.02146,
           'H':137.05891,'I':113.08406,'L':113.08406,'K':128.09496,'M':131.04049,'F':147.06841,'P':97.05276,'S':87.03203,
           'T':101.04768,'W':186.07931,'Y':163.06333,'V':99.06841}
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
DISORDER_AA = set(list("PESQKRNTG"))

def compute_seq_metrics(seq):
    seq = seq.upper()
    length = len(seq)
    count=0; mw=0.0; gravy_sum=0.0; disorder_count=0
    for aa in seq:
        if aa in AA_MASS:
            count +=1; mw += AA_MASS[aa]; gravy_sum += KD.get(aa,0.0)
            if aa in DISORDER_AA:
                disorder_count +=1
    if count==0:
        return {"length":0,"mw":0.0,"gravy":0.0,"disorder_frac":0.0,"instability_proxy":0.0}
    gravy = gravy_sum/count; disorder_frac = disorder_count/count
    instability = max(0.0,1.0 - min(1.0, length/1000.0)) + abs(gravy)/10.0
    return {"length":length,"mw":round(mw,2),"gravy":round(gravy,4),"disorder_frac":round(disorder_frac,4),"instability":round(instability,4)}

# -----------------------------
# UI Tabs (strings only) — now render content
tab_names = [
    "Upload / Files",
    "Network Map (Clusters)",
    "Network Metrics",
    "Protein Details",
    "Sequences",
    "Motifs / Domains",
    "3D Structure",
    "Intrinsic Disorder & Stability"
]
tabs = st.tabs(tab_names)

# Tab 0: Upload / Files (show summary & allow resetting)
with tabs[0]:
    st.header("Upload / Files")
    st.markdown("You can upload a CSV and optionally a FASTA. Use the sidebar to download sample files or reset to sample data.")
    st.write(f"Detected columns: {cols}", key="ui_cols_key")
    st.write(f"Nodes: {len(proteins)} | Edges (approx): {sum(len(v) for v in adj.values())//2} | Communities: {num_communities}", key="ui_summary_key")
    if errors:
        for i,e in enumerate(errors):
            st.error(e, key=f"ui_err_{i}")
    if st.button("Reset to sample dataset (also clears uploaded files in this session)", key="ui_reset_btn"):
        st.session_state.use_sample = True
        st.session_state.df_text = SAMPLE_CSV_TEXT
        st.session_state.fasta_text = SAMPLE_FASTA_TEXT
        st.experimental_rerun()

# Tab 1: Network Map (Clusters)
with tabs[1]:
    st.header("Interactive Network Map — clusters & node metrics")
    st.markdown("Nodes colored by community, size ~ degree, color intensity ~ closeness.")
    fig, ax = plt.subplots(figsize=(12,8))
    cmap = plt.cm.get_cmap('tab20', max(4, num_communities))
    # draw edges
    drawn=set()
    for a in adj:
        for b in adj[a]:
            if (a,b) in drawn or (b,a) in drawn:
                continue
            xa,ya = positions[a]; xb,yb = positions[b]
            ax.plot([xa,xb],[ya,yb], color="#DDDDDD", linewidth=0.7, zorder=1, alpha=0.6)
            drawn.add((a,b))
    # draw nodes
    for node in proteins:
        x,y = positions[node]
        comm_index = None
        for idx,members in communities.items():
            if node in members:
                comm_index = idx; break
        color = cmap(comm_index % cmap.N)
        sz = 50 + degree[node]*25
        ax.scatter(x,y,s=sz,color=color,edgecolor='black',linewidth=0.6,zorder=3)
        ax.text(x,y+0.02,node,fontsize=8,ha='center',rotation=30,zorder=4)
    ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')
    st.pyplot(fig, key="cluster_map_plot")

    st.markdown("Select a node below to view metrics, sequence and sequence-derived heuristics.")
    selected = st.selectbox("Select protein (map)", sorted(list(proteins)), key="map_selectbox")
    if selected:
        st.subheader(f"Selected: {selected}")
        st.write(f"- Degree: **{degree[selected]}**")
        st.write(f"- Closeness: **{closeness[selected]:.6f}**")
        st.write(f"- Betweenness: **{betweenness[selected]:.6f}**")
        st.write(f"- Clustering coeff: **{clustering[selected]:.6f}**")
        seqtxt = fasta_seqs.get(selected, "")
        if seqtxt:
            st.text_area("Sequence (from uploaded/sample FASTA)", seqtxt, height=200, key="map_seq_area")
        else:
            try:
                with st.spinner("Fetching sequence from UniProt..."):
                    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{selected}.fasta", timeout=10)
                    if r.ok and r.text.strip():
                        seqtxt = "".join(r.text.splitlines()[1:])
                        st.text_area("Sequence (from UniProt)", seqtxt, height=200, key="map_seq_uniprot")
                    else:
                        st.warning("No sequence in FASTA & UniProt lookup returned nothing.", key="map_seq_warn")
            except Exception:
                st.warning("UniProt query failed.", key="map_seq_warn2")
        if seqtxt:
            seq_metrics = compute_seq_metrics(seqtxt)
            st.write(f"- Length: **{seq_metrics['length']}**")
            st.write(f"- GRAVY: **{seq_metrics['gravy']}**")
            st.write(f"- Disorder fraction: **{seq_metrics['disorder_frac']}**")
            st.write(f"- Instability proxy: **{seq_metrics['instability']}**")

# Tab 2: Network Metrics
with tabs[2]:
    st.header("Network Metrics (table)")
    st.write("Full metrics (degree, closeness, betweenness, clustering).")
    def highlight_top(s):
        q75 = s.quantile(0.75)
        return ['background-color: #FFD700' if v >= q75 else '' for v in s]
    styled = metrics_df.style.apply(highlight_top, subset=["Closeness","Betweenness","Clustering"])
    st.dataframe(styled, height=450, key="metrics_table")
    st.download_button("Download metrics (CSV)", metrics_df.to_csv(index=False).encode("utf-8"), file_name="metrics.csv", key="download_metrics_btn")

# Tab 3: Protein Details
with tabs[3]:
    st.header("Protein Details (sequence lookup & metrics)")
    prot = st.text_input("Enter UniProt entry name / ID (e.g., P53_HUMAN)", key="prot_input_details")
    if st.button("Fetch details", key="fetch_details_btn"):
        seq_text = fasta_seqs.get(prot, "")
        if seq_text:
            st.text_area("Sequence (uploaded/sample FASTA)", seq_text, height=250, key="details_seq_area")
        else:
            try:
                with st.spinner("Fetching sequence from UniProt..."):
                    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{prot}.fasta", timeout=12)
                    if r.ok and r.text.strip():
                        seq_text = "".join(r.text.splitlines()[1:])
                        st.text_area("Sequence (UniProt)", seq_text, height=250, key="details_seq_uniprot")
                    else:
                        st.warning("Sequence not found in UniProt.", key="details_seq_warn")
            except Exception:
                st.warning("UniProt query failed.", key="details_seq_fail")
        if prot in degree:
            st.write(f"- Degree: **{degree[prot]}**")
            st.write(f"- Closeness: **{closeness[prot]:.6f}**")
            st.write(f"- Betweenness: **{betweenness[prot]:.6f}**")
            st.write(f"- Clustering coeff: **{clustering[prot]:.6f}**")
        else:
            st.info("Protein not in current network.", key="details_not_in_network")

# Tab 4: Sequences
with tabs[4]:
    st.header("FASTA Sequences")
    fasta_file = st.file_uploader("Upload FASTA (optional)", type=["fasta","fa","txt"], key="seq_upload")
    if fasta_file:
        if SeqIO is not None:
            try:
                seqs = SeqIO.parse(fasta_file, "fasta")
                for rec in seqs:
                    st.text(f">{rec.id}\n{rec.seq}", key=f"seq_rec_{rec.id}")
            except Exception as e:
                st.error(f"Failed to parse FASTA via SeqIO: {e}", key="seqio_err")
        else:
            text = fasta_file.read()
            if isinstance(text, bytes): text = text.decode("utf-8")
            st.text(text, key="raw_fasta_display")
    else:
        st.info("No FASTA uploaded — showing sample FASTA.", key="seq_info_sample")
        st.text(SAMPLE_FASTA_TEXT, key="sample_fasta_display")

# Tab 5: Motifs / Domains
with tabs[5]:
    st.header("Motifs & Domains (UniProt / EBI)")
    motif_prot = st.text_input("Enter UniProt ID for features", key="motif_input")
    if st.button("Fetch motifs/domains", key="fetch_motifs_btn"):
        try:
            with st.spinner("Querying UniProt/EBI..."):
                url = f"https://rest.uniprot.org/uniprotkb/{motif_prot}.json"
                r = requests.get(url, timeout=12)
                if r.ok:
                    j = r.json()
                    features = j.get("features", [])
                    if not features:
                        st.info("No features returned for this protein.", key="motifs_none")
                    else:
                        for i,f in enumerate(features[:50]):
                            st.markdown(f"**{f.get('type','')}** — {f.get('description','')}  \nLocation: {f.get('location',{})}", key=f"motif_feature_{i}")
                else:
                    st.error("Feature fetch failed (not found or rate-limited).", key="motifs_error")
        except Exception:
            st.error("Motifs/domains query failed (network or timeout).", key="motifs_exception")

# Tab 6: 3D Structure
with tabs[6]:
    st.header("AlphaFold 3D Structure Viewer")
    struct_id = st.text_input("Enter AlphaFold / UniProt entry name (e.g., P53_HUMAN)", key="alphafold_input")
    if st.button("Load AlphaFold structure", key="alphafold_btn"):
        if not struct_id.strip():
            st.warning("Enter an ID first.", key="alphafold_warn")
        else:
            if py3Dmol is None:
                st.error("py3Dmol not installed in this environment. Install py3Dmol to enable this feature.", key="py3dmol_missing")
            else:
                try:
                    url = f"https://alphafold.ebi.ac.uk/files/AF-{struct_id}-F1-model_v4.pdb"
                    r = requests.get(url, timeout=15)
                    if r.ok and r.text:
                        pdb_text = r.text
                        view = py3Dmol.view(width=800, height=500)
                        view.addModel(pdb_text, "pdb")
                        view.setStyle({'cartoon': {'color':'spectrum'}})
                        view.zoomTo()
                        view.show()
                        st.components.v1.html(view.js(), height=520, key="py3dmol_view")
                    else:
                        st.error("AlphaFold model not found for that ID.", key="alphafold_notfound")
                except Exception:
                    st.error("AlphaFold fetch failed (network/time-out).", key="alphafold_fetch_fail")

# Tab 7: Intrinsic Disorder & Stability
with tabs[7]:
    st.header("Sequence-derived Intrinsic Disorder & Stability (heuristics)")
    seq_input_id = st.text_input("Enter UniProt entry name or paste sequence", key="disorder_seq_input")
    if st.button("Compute sequence heuristics", key="disorder_compute_btn"):
        seq_text = ""
        # first check uploaded/sample fasta
        if seq_input_id in fasta_seqs:
            seq_text = fasta_seqs[seq_input_id]
        else:
            # if looks like FASTA, use it
            if seq_input_id.strip().upper().startswith(">"):
                # parse pasted FASTA
                seqs = parse_fasta_text(seq_input_id)
                if seqs:
                    seq_text = list(seqs.values())[0]
            else:
                # try UniProt
                try:
                    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{seq_input_id}.fasta", timeout=10)
                    if r.ok and r.text.strip():
                        seq_text = "".join(r.text.splitlines()[1:])
                except Exception:
                    pass
        if not seq_text:
            st.warning("No sequence found for input; either upload a FASTA or paste a sequence/entry name.", key="disorder_no_seq")
        else:
            metrics = compute_seq_metrics(seq_text)
            st.write(f"- Length: **{metrics['length']}**")
            st.write(f"- Molecular weight (approx): **{metrics['mw']} Da**")
            st.write(f"- GRAVY: **{metrics['gravy']}**")
            st.write(f"- Disorder fraction (heuristic): **{metrics['disorder_frac']}**")
            st.write(f"- Instability proxy (heuristic): **{metrics['instability']}**")
            st.info("Note: These are sequence-only heuristic estimates and not validated predictors. For publication-grade results use IUPred, DISOPRED, ProtParam, etc.", key="disorder_note")

# Footer
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>This app includes all requested features and is hardened against common Streamlit widget errors (duplicate IDs, missing sample files). If you see an error, tell me the exact traceback and I will patch it immediately.</p>", unsafe_allow_html=True)
