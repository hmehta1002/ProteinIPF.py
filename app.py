# app.py
# Full Protein Network Explorer — All features preserved + unique Streamlit keys + disorder/stability estimates
import streamlit as st
import pandas as pd
import requests
import py3Dmol
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import itertools
import math
import random
import io

# -----------------------------
# Page config & header
st.set_page_config(page_title="Protein Network Explorer — Full", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align:center; color:#4B0082;'>🧬 Protein Network Explorer — Full</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6A5ACD;'>All original features + improved UI/UX + interactive cluster map + disorder/stability estimates</p>", unsafe_allow_html=True)

# -----------------------------
# ---------- Sample data strings (embedded) ----------
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
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWELK
>MDM2_HUMAN
MDM2SEQEXAMPLEXXXXXXXXXXXXXX
>BRCA1_HUMAN
BRCA1SEQEXAMPLEYYYYYYYYYYYYYYY
>BRCA2_HUMAN
BRCA2SEQEXAMPLEZZZZZZZZZZZZZZZ
>RAD51_HUMAN
RAD51SEQEXAMPLEAAAAAAAAAAAAAAA
>UBC_HUMAN
UBCSEQEXAMPLEBBBBBBBBBBBBBBB
>AKT1_HUMAN
AKT1SEQEXAMPLECCCCCCCCCCCCCCC
>MTOR_HUMAN
MTORSEQEXAMPLEDDDDDDDDDDDDDDD
>PIK3CA_HUMAN
PIK3CASEQEXAMPLEEEEEEEEEEEEEE
>PTEN_HUMAN
PTENSEQEXAMPLEFFFFFFFFFFFFFFF
>TP53BP1_HUMAN
TP53BP1SEQEXAMPLEGGGGGGGGGGGGG
"""

# -----------------------------
# Sidebar: uploads, sample downloads, options
st.sidebar.header("Upload / Samples")
uploaded_file = st.sidebar.file_uploader("Upload protein network CSV (edges)", type=["csv"], key="uploader_edges_001")
uploaded_fasta = st.sidebar.file_uploader("(Optional) Upload FASTA file (sequences)", type=["fasta","fa","txt"], key="uploader_fasta_001")

st.sidebar.markdown("**Sample files**")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    st.download_button("Download sample CSV", SAMPLE_CSV.encode('utf-8'), file_name="sample_network.csv", key="download_sample_csv_001")
with col_b:
    st.download_button("Download sample FASTA", SAMPLE_FASTA.encode('utf-8'), file_name="sample_sequences.fasta", key="download_sample_fasta_001")

if st.sidebar.button("Load sample CSV + FASTA into app", key="load_samples_btn_001"):
    uploaded_file = io.StringIO(SAMPLE_CSV)
    uploaded_fasta = io.StringIO(SAMPLE_FASTA)
    st.sidebar.success("Sample CSV & FASTA loaded into app memory (you can still upload your own files).", key="load_samples_success_001")

st.sidebar.markdown("---")
filter_human = st.sidebar.checkbox("Filter for Homo sapiens (TaxID 9606)", value=True, key="filter_human_key_001")
st.sidebar.markdown("If your CSV headers differ, set mapping below:")
with st.sidebar.expander("Column name mapping (if headers differ)", expanded=False):
    c1 = st.text_input("Protein A column", value="Protein1", key="map_protA_001")
    c2 = st.text_input("Protein B column", value="Protein2", key="map_protB_001")
    ctax = st.text_input("TaxID column", value="TaxID", key="map_taxid_001")

# -----------------------------
# Utility: read CSV into pandas and records
def read_csv_to_records(filelike):
    try:
        df = pd.read_csv(filelike)
        return df, df.to_dict('records'), list(df.columns)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}", key="csv_read_error_001")
        return None, None, None

# Load CSV if provided
if uploaded_file:
    df_edges, data_records, columns = read_csv_to_records(uploaded_file)
else:
    df_edges, data_records, columns = None, [], []

# -----------------------------
# Parse FASTA (uploaded or sample)
def parse_fasta(filelike):
    seqs = {}
    try:
        text = filelike.read()
        if isinstance(text, bytes):
            text = text.decode('utf-8')
        entries = text.strip().split('>')
        for e in entries:
            if not e.strip():
                continue
            lines = e.strip().splitlines()
            header = lines[0].split()[0]
            seq = "".join(lines[1:]).replace(" ", "").replace("\r","")
            seqs[header] = seq
    except Exception:
        pass
    return seqs

fasta_seqs = {}
sample_fasta_loaded = False
if uploaded_fasta:
    fasta_seqs = parse_fasta(uploaded_fasta)
    sample_fasta_loaded = True
elif isinstance(uploaded_file, io.StringIO) and uploaded_file is not None and not uploaded_fasta:
    # If user loaded sample via button, also load sample fasta
    fasta_seqs = parse_fasta(io.StringIO(SAMPLE_FASTA))
    sample_fasta_loaded = True

# -----------------------------
# Build adjacency list with flexible headers and error handling
adj = {}
proteins = set()
error_msgs = []

if data_records:
    df_cols = columns
    def choose_col(name, fallbacks):
        if name in df_cols:
            return name
        for f in fallbacks:
            if f in df_cols:
                return f
        return None
    protA_col = choose_col(c1, ["Protein1","protein1","protA","InteractorA","source","geneA","GeneA","protein_A"])
    protB_col = choose_col(c2, ["Protein2","protein2","protB","InteractorB","target","geneB","GeneB","protein_B"])
    tax_col = choose_col(ctax, ["TaxID","taxid","tax_id","Tax_Id"])
    if protA_col is None or protB_col is None:
        error_msgs.append("Could not find protein columns. Detected columns: " + ", ".join(df_cols) + 
                          ". Use the mapping fields in the sidebar to set correct names.")
    else:
        for row in data_records:
            try:
                p1 = str(row.get(protA_col)).strip()
                p2 = str(row.get(protB_col)).strip()
                if filter_human and tax_col:
                    taxv = str(row.get(tax_col))
                    if taxv and taxv not in ["9606","9606.0","9606.00"]:
                        continue
                if p1 and p1.lower() not in ["nan","none"] and p2 and p2.lower() not in ["nan","none"]:
                    proteins.update([p1,p2])
                    adj.setdefault(p1, set()).add(p2)
                    adj.setdefault(p2, set()).add(p1)
            except Exception:
                continue

# If no user data, load sample automatically
if not adj and not data_records:
    df_example = pd.read_csv(io.StringIO(SAMPLE_CSV))
    for _, r in df_example.iterrows():
        p1 = r['Protein1']; p2 = r['Protein2']; tax = r.get('TaxID', None)
        if filter_human and tax and str(tax) != "9606":
            continue
        proteins.update([p1,p2])
        adj.setdefault(p1, set()).add(p2)
        adj.setdefault(p2, set()).add(p1)
    data_records = df_example.to_dict('records')
    if not fasta_seqs:
        fasta_seqs = parse_fasta(io.StringIO(SAMPLE_FASTA))
        sample_fasta_loaded = True

# If still empty, stop with message
if not adj:
    st.warning("No network loaded. Upload a CSV or click 'Load sample CSV + FASTA into app' in the sidebar.", key="no_network_warning_001")
    if error_msgs:
        for i,m in enumerate(error_msgs):
            st.error(m, key=f"csv_error_msg_{i:03d}")
    st.stop()

# -----------------------------
# ---------- Network metrics (pure Python) ----------
def bfs_shortest_paths_local(graph, start):
    visited = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = visited[node] + 1
                queue.append(neighbor)
    return visited

# Closeness
closeness = {}
for node in proteins:
    sp = bfs_shortest_paths_local(adj, node)
    if len(sp) > 1:
        total = sum(sp.values())
        closeness[node] = (len(sp)-1)/total if total > 0 else 0.0
    else:
        closeness[node] = 0.0

# Betweenness (BFS-based accumulation)
betweenness = dict.fromkeys(proteins, 0.0)
for s in proteins:
    S = []
    P = {v: [] for v in proteins}
    sigma = dict.fromkeys(proteins, 0.0)
    dist = dict.fromkeys(proteins, -1)
    sigma[s] = 1.0
    dist[s] = 0
    Q = deque([s])
    while Q:
        v = Q.popleft()
        S.append(v)
        for w in adj[v]:
            if dist[w] < 0:
                dist[w] = dist[v] + 1
                Q.append(w)
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]
                P[w].append(v)
    delta = dict.fromkeys(proteins, 0.0)
    while S:
        w = S.pop()
        for v in P[w]:
            if sigma[w] != 0:
                delta_v = (sigma[v] / sigma[w]) * (1 + delta[w])
            else:
                delta_v = 0
            delta[v] += delta_v
        if w != s:
            betweenness[w] += delta[w]

# Clustering coefficient
clustering = {}
for node in proteins:
    neighbors = list(adj[node])
    k = len(neighbors)
    if k < 2:
        clustering[node] = 0.0
    else:
        links = 0
        for (u,v) in itertools.combinations(neighbors, 2):
            if u in adj[v]:
                links += 1
        clustering[node] = 2 * links / (k * (k - 1))

# Degree
degree = {node: len(adj[node]) for node in proteins}

metrics_df = pd.DataFrame({
    "Protein": list(proteins),
    "Degree": [degree[p] for p in proteins],
    "Closeness": [closeness[p] for p in proteins],
    "Betweenness": [betweenness[p] for p in proteins],
    "Clustering": [clustering[p] for p in proteins]
}).sort_values(by="Degree", ascending=False).reset_index(drop=True)

# -----------------------------
# ---------- Label Propagation for communities ----------
def label_propagation(graph, max_iter=200):
    labels = {n: n for n in graph}
    nodes = list(graph.keys())
    for it in range(max_iter):
        changed = False
        random.shuffle(nodes)
        for node in nodes:
            counts = defaultdict(int)
            for nbr in graph[node]:
                counts[labels[nbr]] += 1
            if not counts:
                continue
            maxc = max(counts.values())
            choices = [lab for lab,c in counts.items() if c == maxc]
            new_label = random.choice(choices)
            if labels[node] != new_label:
                labels[node] = new_label
                changed = True
        if not changed:
            break
    unique = {}
    communities = defaultdict(list)
    idx = 0
    for n,lbl in labels.items():
        if lbl not in unique:
            unique[lbl] = idx
            idx += 1
        communities[unique[lbl]].append(n)
    return labels, dict(communities)

labels, communities = label_propagation(adj, max_iter=200)
num_communities = len(communities)

# -----------------------------
# ---------- Spring layout (force-directed) ----------
def spring_layout(graph, iterations=200, width=1.0, height=1.0, k=None):
    nodes = list(graph.keys())
    N = len(nodes)
    if k is None:
        k = math.sqrt((width*height)/max(1, N))
    pos = {n: [random.uniform(0, width), random.uniform(0, height)] for n in nodes}
    t = width / 10.0
    dt = t / (iterations + 1)
    def repulsive_force(d, k):
        if d == 0: return k*k
        return (k*k) / d
    def attractive_force(d, k):
        return (d*d) / k
    for i in range(iterations):
        disp = {n: [0.0, 0.0] for n in nodes}
        for a in nodes:
            for b in nodes:
                if a == b:
                    continue
                dx = pos[a][0] - pos[b][0]
                dy = pos[a][1] - pos[b][1]
                dist = math.hypot(dx, dy) + 1e-9
                f = repulsive_force(dist, k)
                disp[a][0] += (dx / dist) * f
                disp[a][1] += (dy / dist) * f
        for a in nodes:
            for b in graph[a]:
                dx = pos[a][0] - pos[b][0]
                dy = pos[a][1] - pos[b][1]
                dist = math.hypot(dx, dy) + 1e-9
                f = attractive_force(dist, k)
                disp[a][0] -= (dx / dist) * f
                disp[a][1] -= (dy / dist) * f
        for n in nodes:
            dx, dy = disp[n]
            disp_len = math.hypot(dx, dy)
            if disp_len > 0:
                pos[n][0] += (dx / disp_len) * min(disp_len, t)
                pos[n][1] += (dy / disp_len) * min(disp_len, t)
            pos[n][0] = min(width, max(0.0, pos[n][0]))
            pos[n][1] = min(height, max(0.0, pos[n][1]))
        t -= dt
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    for n in nodes:
        if maxx-minx > 0:
            pos[n][0] = (pos[n][0] - minx) / (maxx - minx)
        else:
            pos[n][0] = 0.5
        if maxy-miny > 0:
            pos[n][1] = (pos[n][1] - miny) / (maxy - miny)
        else:
            pos[n][1] = 0.5
    return pos

with st.spinner("Computing layout..."):
    positions = spring_layout(adj, iterations=250, width=1.0, height=1.0, k=None)

# -----------------------------
# ---------- Sequence-based disorder & stability estimates ----------
# NOTE: Simple sequence-derived metrics (GRAVY, length, molecular weight approx, disorder-promoting residue fraction)
AA_MASSES = {
    'A': 71.03711,'R':156.10111,'N':114.04293,'D':115.02694,'C':103.00919,'E':129.04259,'Q':128.05858,'G':57.02146,
    'H':137.05891,'I':113.08406,'L':113.08406,'K':128.09496,'M':131.04049,'F':147.06841,'P':97.05276,'S':87.03203,
    'T':101.04768,'W':186.07931,'Y':163.06333,'V':99.06841
}
KD_SCALE = {
 'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,
 'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2
}
DISORDER_PROMOTING = set(list("PESQKRNTG"))  # simple approximation (residues often enriched in disorder)

def compute_sequence_metrics(seq):
    seq = seq.upper()
    length = len(seq)
    mw = 0.0
    gravy = 0.0
    count_valid = 0
    disorder_count = 0
    for aa in seq:
        if aa in AA_MASSES:
            mw += AA_MASSES[aa]
            gravy += KD_SCALE.get(aa,0.0)
            count_valid += 1
            if aa in DISORDER_PROMOTING:
                disorder_count += 1
    mw = mw if count_valid>0 else 0.0
    gravy_avg = gravy/count_valid if count_valid>0 else 0.0
    disorder_frac = disorder_count/count_valid if count_valid>0 else 0.0
    # rudimentary "instability" proxy: shorter proteins + extreme gravy may be more/less stable — we present as a simple score
    instability_score = max(0.0, 1.0 - min(1.0, length/1000.0)) + abs(gravy_avg)/10.0  # arbitrary combination, for guidance only
    return {
        "length": length,
        "molecular_weight": round(mw,2),
        "gravy": round(gravy_avg,4),
        "disorder_fraction": round(disorder_frac,4),
        "instability_proxy": round(instability_score,4)
    }

# -----------------------------
# ---------- UI Tabs (all features preserved) ----------
tabs = st.tabs(["Upload/Files", "Network Map (Clusters)", "Network Metrics", "Protein Details", "3D Viewer", "Motifs/Domains", "Downloads"], key="main_tabs_001")

# Tab 0: Upload/Files
with tabs[0]:
    st.header("Upload / Sample Files")
    st.markdown("You can upload your own CSV (edges) and optionally a FASTA. Or use the sample files in the sidebar.")
    st.markdown("**Detected columns (if you uploaded):**")
    if columns:
        st.write(columns, key="detected_columns_001")
    else:
        st.write("No CSV uploaded — sample dataset will be used.", key="no_columns_text_001")
    st.markdown("**Quick data summary:**")
    st.write(f"Number of proteins (nodes): **{len(proteins)}**", key="summary_nodes_001")
    st.write(f"Number of edges (approx): **{sum(len(v) for v in adj.values())//2}**", key="summary_edges_001")
    st.write(f"Number of detected communities: **{num_communities}**", key="summary_comms_001")
    if error_msgs:
        for i,m in enumerate(error_msgs):
            st.error(m, key=f"upload_error_{i:03d}")

# Tab 1: Network Map (Clusters)
with tabs[1]:
    st.header("Interactive Network Map — Clusters & Node Metrics")
    st.markdown("Nodes placed by spring layout; colors = community, size = degree, intensity ~ closeness.")
    fig, ax = plt.subplots(figsize=(12,8))
    cmap = plt.cm.get_cmap('tab20', max(4, num_communities))
    # draw edges
    drawn = set()
    for a in adj:
        for b in adj[a]:
            if (b,a) in drawn or (a,b) in drawn:
                continue
            xa, ya = positions[a]; xb, yb = positions[b]
            ax.plot([xa, xb], [ya, yb], color="#DDDDDD", linewidth=0.8, zorder=1, alpha=0.6)
            drawn.add((a,b))
    # draw nodes
    for node in proteins:
        x,y = positions[node]
        # find community id
        comm_index = None
        for idx, members in communities.items():
            if node in members:
                comm_index = idx
                break
        color = cmap(comm_index % cmap.N)
        sz = 50 + degree[node] * 25
        clos_vals = list(closeness.values())
        vmax = max(clos_vals) if clos_vals else 1
        vmin = min(clos_vals) if clos_vals else 0
        norm_val = (closeness[node] - vmin) / (vmax - vmin + 1e-12)
        edgecolor = 'black'
        ax.scatter(x, y, s=sz, color=color, edgecolor=edgecolor, linewidth=0.6, zorder=3)
        ax.text(x, y+0.02, node, fontsize=8, ha='center', rotation=30, zorder=4)
    ax.set_title("Cluster Map — colored by community; node size ~ degree", color="#4B0082", fontsize=14)
    ax.set_xticks([]); ax.set_yticks([]); ax.axis('off')
    st.pyplot(fig, key="cluster_map_plot_001")

    st.markdown("**Interact with the map below:** select a node to view metrics + sequence + disorder/stability estimates.")
    selected_from_map = st.selectbox("Select a protein (map)", sorted(list(proteins)), key="select_map_protein_001")
    if selected_from_map:
        st.markdown(f"### Selected: **{selected_from_map}**")
        st.write(f"- Degree: **{degree[selected_from_map]}**")
        st.write(f"- Closeness: **{closeness[selected_from_map]:.6f}**")
        st.write(f"- Betweenness: **{betweenness[selected_from_map]:.6f}**")
        st.write(f"- Clustering coeff: **{clustering[selected_from_map]:.6f}**")
        # sequence from uploaded FASTA or sample or UniProt fallback
        seq_text = fasta_seqs.get(selected_from_map, "")
        if seq_text:
            st.text_area("Sequence (from uploaded/sample FASTA)", seq_text, height=200, key="seq_textarea_map_001")
        else:
            with st.spinner("Fetching sequence from UniProt..."):
                try:
                    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{selected_from_map}.fasta", timeout=10)
                    if r.ok and r.text.strip():
                        seq_text = "".join(r.text.splitlines()[1:])
                        st.text_area("Sequence (from UniProt)", seq_text, height=200, key="seq_textarea_map_002")
                    else:
                        st.warning("Sequence not found in UniProt and not present in uploaded FASTA.", key="seq_warn_map_001")
                except Exception:
                    st.warning("UniProt query failed (network or rate limit).", key="seq_warn_map_002")
        # compute and show sequence metrics if available
        if seq_text:
            seq_metrics = compute_sequence_metrics(seq_text)
            st.subheader("Sequence-derived estimates")
            st.write(f"- Length: **{seq_metrics['length']}**")
            st.write(f"- Approx. molecular weight (Da): **{seq_metrics['molecular_weight']}**")
            st.write(f"- GRAVY (hydropathy average): **{seq_metrics['gravy']}**")
            st.write(f"- Disorder-promoting residue fraction (simple estimate): **{seq_metrics['disorder_fraction']}**")
            st.write(f"- Instability proxy (simple heuristic): **{seq_metrics['instability_proxy']}**")
            st.info("Note: These are sequence-only, heuristic estimates (not a validated predictor). Use specialised tools (IUPred, ProtParam) for publication-grade analysis.", icon="ℹ️", key="seq_info_note_001")

# Tab 2: Network Metrics
with tabs[2]:
    st.header("Network Metrics (table)")
    st.markdown("Full metrics table — sort and download. Top values highlighted.")
    def highlight_top_quartile(s):
        q75 = s.quantile(0.75)
        return ['background-color: #FFD700' if v >= q75 else '' for v in s]
    styled = metrics_df.style.apply(highlight_top_quartile, subset=["Closeness","Betweenness","Clustering"])
    st.dataframe(styled, height=450, key="metrics_table_df_001")
    csv_bytes = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download metrics table (CSV)", csv_bytes, file_name="protein_metrics.csv", key="download_metrics_csv_001")

# Tab 3: Protein Details
with tabs[3]:
    st.header("Protein Details — Sequence & Basic Metrics")
    select_prot = st.selectbox("Choose a protein", sorted(list(proteins)), key="select_protein_details_001")
    if select_prot:
        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Sequence")
            seq_text = fasta_seqs.get(select_prot, "")
            if seq_text:
                st.text_area("Sequence (from uploaded/sample FASTA)", seq_text, height=300, key="seq_textarea_details_001")
            else:
                st.info("No uploaded FASTA sequence; attempting UniProt lookup.", key="uniprot_info_001")
                try:
                    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{select_prot}.fasta", timeout=10)
                    if r.ok and r.text.strip():
                        seq_text = "".join(r.text.splitlines()[1:])
                        st.text_area("Sequence (from UniProt)", seq_text, height=300, key="seq_textarea_details_002")
                    else:
                        st.warning("UniProt returned no sequence for this ID.", key="uniprot_warn_001")
                except Exception:
                    st.warning("UniProt query failed.", key="uniprot_fail_001")
        with col2:
            st.subheader("Basic Metrics")
            st.write(f"- Degree: **{degree[select_prot]}**", key="bd_degree_001")
            st.write(f"- Closeness: **{closeness[select_prot]:.6f}**", key="bd_closeness_001")
            st.write(f"- Betweenness: **{betweenness[select_prot]:.6f}**", key="bd_betweenness_001")
            st.write(f"- Clustering coeff: **{clustering[select_prot]:.6f}**", key="bd_clustering_001")
            st.markdown("**AlphaFold structure**: try the 3D Viewer tab with the entry name (e.g., P53_HUMAN).", key="alphafold_note_001")
            # sequence metrics if available
            if seq_text:
                seq_metrics = compute_sequence_metrics(seq_text)
                st.markdown("**Sequence-derived estimates**", key="seq_estimates_header_001")
                st.write(f"- Length: **{seq_metrics['length']}**", key="seq_len_001")
                st.write(f"- GRAVY: **{seq_metrics['gravy']}**", key="seq_gravy_001")
                st.write(f"- Disorder fraction: **{seq_metrics['disorder_fraction']}**", key="seq_disorder_001")
                st.write(f"- Instability proxy: **{seq_metrics['instability_proxy']}**", key="seq_instab_001")

# Tab 4: 3D Viewer
with tabs[4]:
    st.header("AlphaFold 3D Structure Viewer")
    st.markdown("Enter an AlphaFold ID (UniProt entry name) and click Load.")
    structure_id = st.text_input("AlphaFold ID (e.g., P53_HUMAN)", value="", key="alphafold_input_001")
    if st.button("Load AlphaFold structure", key="alphafold_load_btn_001"):
        if not structure_id.strip():
            st.warning("Enter a structure ID first.", key="alphafold_enter_warn_001")
        else:
            with st.spinner("Fetching AlphaFold PDB..."):
                url = f"https://alphafold.ebi.ac.uk/files/AF-{structure_id}-F1-model_v4.pdb"
                try:
                    r = requests.get(url, timeout=15)
                    if r.ok and r.text:
                        pdb_text = r.text
                        view = py3Dmol.view(width=800, height=500)
                        view.addModel(pdb_text, "pdb")
                        view.setStyle({'cartoon': {'color':'spectrum'}})
                        view.zoomTo()
                        view.show()
                        st.components.v1.html(view.js(), height=520, key="py3dmol_view_001")
                    else:
                        st.error("Structure not found on AlphaFold (check the ID).", key="alphafold_notfound_001")
                except Exception:
                    st.error("Failed to fetch AlphaFold structure (timeout or network).", key="alphafold_fetch_fail_001")

# Tab 5: Motifs/Domains
with tabs[5]:
    st.header("Motifs & Domains (EBI Proteins API)")
    prot_for_motifs = st.selectbox("Protein to fetch motifs/domains", sorted(list(proteins)), key="motifs_select_001")
    if st.button("Fetch motifs/domains from EBI", key="fetch_motifs_btn_001"):
        with st.spinner("Querying EBI Proteins API..."):
            url = f"https://www.ebi.ac.uk/proteins/api/proteins/{prot_for_motifs}"
            headers = {"Accept":"application/json"}
            try:
                r = requests.get(url, headers=headers, timeout=12)
                if r.ok:
                    info = r.json()
                    features = info.get("features", [])
                    if not features:
                        st.info("No features returned by API for this protein.", key="motifs_none_001")
                    else:
                        for idx,f in enumerate(features):
                            ftype = f.get("type", "feature")
                            desc = f.get("description", "")
                            start = f.get("begin", "")
                            end = f.get("end", "")
                            st.markdown(f"**{ftype}** — {desc}  \nLocation: {start} — {end}", unsafe_allow_html=True, key=f"motif_feature_{idx:03d}_001")
                else:
                    st.error("No information found or API rate-limited.", key="motifs_error_001")
            except Exception:
                st.error("Failed to fetch motifs/domains (network or timeout).", key="motifs_exception_001")

# Tab 6: Downloads
with tabs[6]:
    st.header("Downloads & Exports")
    st.markdown("Download sample files, metrics, or selected protein data.")
    st.download_button("Download sample CSV", SAMPLE_CSV.encode('utf-8'), file_name="sample_network.csv", key="download_sample_csv_002")
    st.download_button("Download sample FASTA", SAMPLE_FASTA.encode('utf-8'), file_name="sample_sequences.fasta", key="download_sample_fasta_002")
    st.download_button("Download metrics table (CSV)", metrics_df.to_csv(index=False).encode('utf-8'), file_name="protein_metrics.csv", key="download_metrics_csv_002")
    sel = st.selectbox("Select protein to export", sorted(list(proteins)), key="export_select_001")
    if st.button("Export selected protein data", key="export_protein_btn_001"):
        seq_text = fasta_seqs.get(sel, "")
        if not seq_text:
            try:
                r = requests.get(f"https://rest.uniprot.org/uniprotkb/{sel}.fasta", timeout=6)
                if r.ok and r.text.strip():
                    seq_text = "".join(r.text.splitlines()[1:])
            except Exception:
                seq_text = ""
        export_dict = {
            "Protein": sel,
            "Degree": degree.get(sel,0),
            "Closeness": closeness.get(sel,0.0),
            "Betweenness": betweenness.get(sel,0.0),
            "Clustering": clustering.get(sel,0.0),
            "Sequence": seq_text
        }
        out_bytes = str(export_dict).encode('utf-8')
        st.download_button("Download selected protein data (txt)", out_bytes, file_name=f"{sel}_data.txt", key="download_export_selected_001")

# Footer note
st.markdown("<hr><p style='text-align:center; color:gray;'>All features preserved: original metrics, static & interactive maps, clustering, sequence fetch, motif/domain fetch, AlphaFold viewer, sample downloads, plus sequence-derived disorder/stability estimates (heuristic).</p>", unsafe_allow_html=True)


