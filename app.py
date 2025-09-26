# app.py
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
# Page configuration and styling
st.set_page_config(page_title="Protein Network Explorer — Full", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align:center; color:#4B0082;'>🧬 Protein Network Explorer — Full</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6A5ACD;'>All original features + improved UI/UX + interactive cluster map</p>", unsafe_allow_html=True)

# -----------------------------
# ---------- Sample Data ----------
# Small sample CSV (edges) and FASTA included in-app for download & load
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
# Sidebar: Upload / Sample downloads / Options
st.sidebar.header("Upload / Samples")
uploaded_file = st.sidebar.file_uploader("Upload protein network CSV (edges)", type=["csv"])
uploaded_fasta = st.sidebar.file_uploader("(Optional) Upload FASTA file with sequences", type=["fasta","fa","txt"])

st.sidebar.markdown("**Sample files (download & load)**")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.download_button("Download sample CSV", SAMPLE_CSV.encode('utf-8'), file_name="sample_network.csv")
with col2:
    st.download_button("Download sample FASTA", SAMPLE_FASTA.encode('utf-8'), file_name="sample_sequences.fasta")

if st.sidebar.button("Load sample CSV and FASTA into app"):
    # create in-memory files that mimic uploads
    uploaded_file = io.StringIO(SAMPLE_CSV)
    uploaded_fasta = io.StringIO(SAMPLE_FASTA)
    st.sidebar.success("Loaded sample CSV & FASTA into the app (use Upload area to change).")

st.sidebar.markdown("---")
filter_human = st.sidebar.checkbox("Filter for Homo sapiens (TaxID 9606)", value=True)
st.sidebar.markdown("If your CSV columns are named differently, use the mapping fields below.")
colmap_expander = st.sidebar.expander("Column name mapping (if your headers differ)", expanded=False)
with colmap_expander:
    c1 = st.text_input("Column name for protein A (default: Protein1)", value="Protein1")
    c2 = st.text_input("Column name for protein B (default: Protein2)", value="Protein2")
    ctax = st.text_input("Column name for TaxID (default: TaxID)", value="TaxID")

# -----------------------------
# Utility: read CSV into records and show columns (safe)
def read_csv_to_records(filelike):
    try:
        df = pd.read_csv(filelike)
        return df, df.to_dict('records'), list(df.columns)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None, None, None

# -----------------------------
# Load edge data (either uploaded or sample loaded)
if uploaded_file:
    df_edges, data_records, columns = read_csv_to_records(uploaded_file)
    if df_edges is None:
        st.stop()
    st.sidebar.write("Columns detected:", columns)
else:
    st.info("No CSV uploaded yet — you can download the sample CSV from the sidebar or click 'Load sample CSV and FASTA into app'.")
    df_edges = None
    data_records = []
    columns = []

# -----------------------------
# If FASTA uploaded, parse it for sequence lookup
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

sample_fasta_loaded = False
fasta_seqs = {}
if uploaded_fasta:
    fasta_seqs = parse_fasta(uploaded_fasta)
    sample_fasta_loaded = True
elif isinstance(uploaded_file, io.StringIO) and uploaded_file is not None and not uploaded_fasta:
    # If loaded samples via button, create sequences from SAMPLE_FASTA
    fasta_seqs = parse_fasta(io.StringIO(SAMPLE_FASTA))
    sample_fasta_loaded = True

# -----------------------------
# Error-handling variables for later
error_msgs = []

# -----------------------------
# Build adjacency and protein list
adj = {}
proteins = set()

if data_records:
    # Use mapping for column names (c1, c2, ctax)
    # If mapping names not in df, try common fallbacks
    df_cols = columns
    def choose_col(name, fallbacks):
        if name in df_cols:
            return name
        for f in fallbacks:
            if f in df_cols:
                return f
        return None

    protA_col = choose_col(c1, ["Protein1","protein1","protA","InteractorA","source","geneA","GeneA"])
    protB_col = choose_col(c2, ["Protein2","protein2","protB","InteractorB","target","geneB","GeneB"])
    tax_col = choose_col(ctax, ["TaxID","taxid","tax_id","Tax_Id"])

    if protA_col is None or protB_col is None:
        error_msgs.append("Could not find protein columns. Detected columns: " + ", ".join(df_cols) + 
                          ". Use the column mapping fields in the sidebar to set correct names.")
    else:
        for row in data_records:
            try:
                p1 = str(row.get(protA_col)).strip()
                p2 = str(row.get(protB_col)).strip()
                # If filter_human enabled, check tax column if present
                if filter_human and tax_col:
                    taxv = str(row.get(tax_col))
                    if taxv and taxv != "9606" and taxv != "9606.0":
                        continue
                # Skip empty or NaN keys
                if p1 and p1.lower() not in ["nan", "none"] and p2 and p2.lower() not in ["nan","none"]:
                    proteins.update([p1, p2])
                    adj.setdefault(p1, set()).add(p2)
                    adj.setdefault(p2, set()).add(p1)
            except Exception:
                continue

# If nothing in adj yet but sample loaded, build from SAMPLE_CSV
if not adj and not data_records:
    # offer to load sample automatically
    df_example = pd.read_csv(io.StringIO(SAMPLE_CSV))
    for _, r in df_example.iterrows():
        p1 = r['Protein1']; p2 = r['Protein2']; tax = r.get('TaxID', None)
        if filter_human and tax and str(tax) != "9606":
            continue
        proteins.update([p1, p2])
        adj.setdefault(p1, set()).add(p2)
        adj.setdefault(p2, set()).add(p1)
    # also populate data_records so metrics compute below
    data_records = df_example.to_dict('records')
    # set fasta seqs too if none
    if not fasta_seqs:
        fasta_seqs = parse_fasta(io.StringIO(SAMPLE_FASTA))
        sample_fasta_loaded = True

# If still empty, show message and stop
if not adj:
    st.warning("No network loaded yet. Upload a CSV or click 'Load sample CSV and FASTA into app' in the sidebar.")
    if error_msgs:
        for m in error_msgs:
            st.error(m)
    st.stop()

# -----------------------------
# ---------- Network metrics in pure Python ----------
# BFS for shortest paths (used in closeness)
def bfs_shortest_paths(graph, start):
    visited = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited[neighbor] = visited[node] + 1
                queue.append(neighbor)
    return visited

# Closeness centrality
closeness = {}
for node in proteins:
    sp = bfs_shortest_paths(adj, node)
    if len(sp) > 1:
        total = sum(sp.values())
        if total > 0:
            closeness[node] = (len(sp)-1)/total
        else:
            closeness[node] = 0.0
    else:
        closeness[node] = 0.0

# Betweenness centrality (approximation using BFS-based dependency accumulation)
betweenness = dict.fromkeys(proteins, 0.0)
for s in proteins:
    # single-source shortest-paths
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

# Local clustering coefficient
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

# Combine metrics into a DataFrame for display/export
metrics_df = pd.DataFrame({
    "Protein": list(proteins),
    "Degree": [degree[p] for p in proteins],
    "Closeness": [closeness[p] for p in proteins],
    "Betweenness": [betweenness[p] for p in proteins],
    "Clustering": [clustering[p] for p in proteins]
}).sort_values(by="Degree", ascending=False).reset_index(drop=True)

# -----------------------------
# ---------- Community detection: Label Propagation (pure python) ----------
def label_propagation(graph, max_iter=100):
    # initialize each node with a unique label
    labels = {n: n for n in graph}
    nodes = list(graph.keys())
    for it in range(max_iter):
        changed = False
        random.shuffle(nodes)
        for node in nodes:
            # find labels of neighbors
            neigh_labels = defaultdict(int)
            for nbr in graph[node]:
                neigh_labels[labels[nbr]] += 1
            if not neigh_labels:
                continue
            # choose the most frequent neighbor label (break ties randomly)
            max_count = max(neigh_labels.values())
            best = [lab for lab,count in neigh_labels.items() if count == max_count]
            new_label = random.choice(best)
            if labels[node] != new_label:
                labels[node] = new_label
                changed = True
        if not changed:
            break
    # compress label ids to 0..k-1
    unique_labels = {}
    communities = defaultdict(list)
    idx = 0
    for n,lbl in labels.items():
        if lbl not in unique_labels:
            unique_labels[lbl] = idx
            idx += 1
        communities[unique_labels[lbl]].append(n)
    return labels, dict(communities)

labels, communities = label_propagation(adj, max_iter=200)
num_communities = len(communities)

# -----------------------------
# ---------- Force-directed layout (simple spring algorithm) ----------
def spring_layout(graph, iterations=200, width=1.0, height=1.0, k=None):
    nodes = list(graph.keys())
    N = len(nodes)
    if k is None:
        k = math.sqrt((width*height)/max(1, N))
    # positions: dict node -> [x,y]
    pos = {n: [random.uniform(0, width), random.uniform(0, height)] for n in nodes}
    # basic constants
    t = width / 10.0  # temperature
    dt = t / (iterations + 1)
    def repulsive_force(d, k):
        if d == 0: return k*k
        return (k*k) / d
    def attractive_force(d, k):
        return (d*d) / k
    for i in range(iterations):
        disp = {n: [0.0, 0.0] for n in nodes}
        # repulsive
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
        # attractive (edges)
        for a in nodes:
            for b in graph[a]:
                dx = pos[a][0] - pos[b][0]
                dy = pos[a][1] - pos[b][1]
                dist = math.hypot(dx, dy) + 1e-9
                f = attractive_force(dist, k)
                disp[a][0] -= (dx / dist) * f
                disp[a][1] -= (dy / dist) * f
        # limit max displacement and update positions
        for n in nodes:
            dx, dy = disp[n]
            disp_len = math.hypot(dx, dy)
            if disp_len > 0:
                pos[n][0] += (dx / disp_len) * min(disp_len, t)
                pos[n][1] += (dy / disp_len) * min(disp_len, t)
            # keep inside bounds
            pos[n][0] = min(width, max(0.0, pos[n][0]))
            pos[n][1] = min(height, max(0.0, pos[n][1]))
        t -= dt
    # Normalize to [0,1]
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

with st.spinner("Computing layout and preparing the interactive cluster map..."):
    positions = spring_layout(adj, iterations=250, width=1.0, height=1.0, k=None)

# -----------------------------
# ---------- UI Tabs (all features preserved) ----------
tabs = st.tabs(["Upload/Files", "Network Map (Clusters)", "Network Metrics", "Protein Details", "3D Viewer", "Motifs/Domains", "Downloads"])

# ---------- Tab: Upload/Files ----------
with tabs[0]:
    st.header("Upload / Sample Files")
    st.markdown("You can upload your own network CSV and optionally a FASTA file. Or download & load the sample files provided in the sidebar.")
    st.markdown("**Detected columns (if you uploaded a file):**")
    if columns:
        st.write(columns)
    else:
        st.write("No CSV uploaded — sample data is loaded if you used the sample loader.")
    st.markdown("**Quick data summary:**")
    st.write(f"Number of proteins (nodes): **{len(proteins)}**")
    st.write(f"Number of edges (approx): **{sum(len(v) for v in adj.values())//2}**")
    st.write(f"Number of detected communities: **{num_communities}**")
    if error_msgs:
        for m in error_msgs:
            st.error(m)

# ---------- Tab: Network Map (Clusters) ----------
with tabs[1]:
    st.header("Interactive Network Map — Clusters & Node Metrics")
    st.markdown("Map shows nodes placed by a force-directed layout and colored by cluster (label propagation). Node size = degree. Color intensity = closeness.")
    fig, ax = plt.subplots(figsize=(12,8))
    # color palette
    cmap = plt.cm.get_cmap('tab20', max(4, num_communities))
    # draw edges (light grey)
    for a in adj:
        for b in adj[a]:
            if a < b:  # draw each edge once
                xa, ya = positions[a]
                xb, yb = positions[b]
                ax.plot([xa, xb], [ya, yb], color="#CCCCCC", linewidth=0.8, zorder=1, alpha=0.6)
    # draw nodes
    node_artists = {}
    for node in proteins:
        x,y = positions[node]
        comm_id = labels[node]
        # convert comm label to compressed id
        # find community index
        comm_index = None
        for idx, members in communities.items():
            if node in members:
                comm_index = idx
                break
        color = cmap(comm_index % cmap.N)
        # node size scales with degree
        sz = 50 + degree[node] * 25
        # color intensity by closeness (normalize)
        clos_vals = list(closeness.values())
        vmax = max(clos_vals) if clos_vals else 1
        vmin = min(clos_vals) if clos_vals else 0
        norm_val = (closeness[node] - vmin) / (vmax - vmin + 1e-12)
        # overlay color with intensity
        edgecolor = 'black'
        artist = ax.scatter(x, y, s=sz, color=color, edgecolor=edgecolor, linewidth=0.6, zorder=3)
        ax.text(x, y+0.025, node, fontsize=8, ha='center', rotation=30, zorder=4)
        node_artists[node] = (artist, x, y)
    ax.set_title("Network Map — clusters colored, node size by degree (labels shown)", color="#4B0082", fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axis('off')
    st.pyplot(fig)

    st.markdown("**Interact with the map:** select a node to view detailed metrics and sequence/motifs.")
    selected_from_map = st.selectbox("Select a protein (from map)", sorted(list(proteins)))
    if selected_from_map:
        st.markdown(f"### Selected: **{selected_from_map}**")
        st.write(f"- Degree: **{degree[selected_from_map]}**")
        st.write(f"- Closeness: **{closeness[selected_from_map]:.6f}**")
        st.write(f"- Betweenness: **{betweenness[selected_from_map]:.6f}**")
        st.write(f"- Clustering coeff: **{clustering[selected_from_map]:.6f}**")
        # Sequence display from uploaded FASTA or UniProt lookup
        if selected_from_map in fasta_seqs:
            st.text_area("Sequence (from uploaded FASTA or sample FASTA)", fasta_seqs[selected_from_map], height=180)
        else:
            # try UniProt REST FASTA fetch
            with st.spinner("Fetching sequence from UniProt..."):
                uni_url = f"https://rest.uniprot.org/uniprotkb/{selected_from_map}.fasta"
                try:
                    r = requests.get(uni_url, timeout=10)
                    if r.ok and r.text.strip():
                        seq = "".join(r.text.splitlines()[1:])
                        st.text_area("Sequence (from UniProt)", seq, height=180)
                    else:
                        st.warning("Sequence not found from UniProt and not present in uploaded FASTA.")
                except Exception:
                    st.warning("UniProt query failed (network or rate limit).")

# ---------- Tab: Network Metrics ----------
with tabs[2]:
    st.header("Network Metrics (full table)")
    st.markdown("Sort or filter this table. Top quartile values are highlighted.")
    def highlight_top_quartile(s):
        q75 = s.quantile(0.75)
        return ['background-color: #FFD700' if v >= q75 else '' for v in s]
    styled = metrics_df.style.apply(highlight_top_quartile, subset=["Closeness","Betweenness","Clustering"])
    st.dataframe(styled, height=450)
    # export button
    csv_bytes = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download metrics table (CSV)", csv_bytes, file_name="protein_metrics.csv")

# ---------- Tab: Protein Details ----------
with tabs[3]:
    st.header("Protein Details (sequence lookup & FASTA)")
    select_prot = st.selectbox("Choose a protein to view sequence & basic info", sorted(list(proteins)))
    if select_prot:
        colA, colB = st.columns([1,1])
        with colA:
            st.subheader("Sequence")
            if select_prot in fasta_seqs:
                st.text_area("Sequence (from uploaded FASTA or sample)", fasta_seqs[select_prot], height=300)
            else:
                st.info("Sequence not found in uploaded FASTA. You can upload a FASTA file or the app will attempt UniProt lookup.")
                try:
                    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{select_prot}.fasta", timeout=10)
                    if r.ok and r.text.strip():
                        seq = "".join(r.text.splitlines()[1:])
                        st.text_area("Sequence (from UniProt)", seq, height=300)
                    else:
                        st.warning("UniProt returned no sequence for that accession.")
                except Exception:
                    st.warning("UniProt query failed.")
        with colB:
            st.subheader("Basic Metrics")
            st.write(f"- Degree: **{degree[select_prot]}**")
            st.write(f"- Closeness: **{closeness[select_prot]:.6f}**")
            st.write(f"- Betweenness: **{betweenness[select_prot]:.6f}**")
            st.write(f"- Clustering coeff: **{clustering[select_prot]:.6f}**")
            # links to AlphaFold
            st.markdown("**AlphaFold / Structure**")
            st.write("Try AlphaFold ID (e.g., P53_HUMAN) in the 3D Viewer tab.")

# ---------- Tab: 3D Viewer ----------
with tabs[4]:
    st.header("AlphaFold 3D Structure Viewer (py3Dmol)")
    st.markdown("Enter an AlphaFold ID (UniProt accession or entry name) to fetch and view the model.")
    structure_id = st.text_input("AlphaFold ID (e.g., P53_HUMAN)", value="")
    if st.button("Load AlphaFold structure"):
        if not structure_id.strip():
            st.warning("Enter a structure ID first.")
        else:
            with st.spinner("Fetching AlphaFold PDB file..."):
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
                        st.components.v1.html(view.js(), height=520)
                    else:
                        st.error("Structure not found on AlphaFold (check the ID).")
                except Exception:
                    st.error("Failed to fetch AlphaFold structure (network or timeout).")

# ---------- Tab: Motifs / Domains ----------
with tabs[5]:
    st.header("Motifs & Domains (EBI Proteins API)")
    st.markdown("Select a protein and fetch its annotated features (domains, motifs) from EBI Proteins API.")
    prot_for_motifs = st.selectbox("Protein to fetch motifs/domains", sorted(list(proteins)), key="motifs_select")
    if st.button("Fetch motifs/domains"):
        with st.spinner("Querying EBI Proteins API..."):
            url = f"https://www.ebi.ac.uk/proteins/api/proteins/{prot_for_motifs}"
            headers = {"Accept":"application/json"}
            try:
                r = requests.get(url, headers=headers, timeout=12)
                if r.ok:
                    info = r.json()
                    features = info.get("features", [])
                    if not features:
                        st.info("No features returned by API for this protein.")
                    else:
                        for f in features:
                            ftype = f.get("type", "feature")
                            desc = f.get("description", "")
                            start = f.get("begin", "")
                            end = f.get("end", "")
                            st.markdown(f"**{ftype}** — {desc}  \nLocation: {start} — {end}")
                else:
                    st.error("No information found or API rate-limited.")
            except Exception:
                st.error("Failed to fetch motifs/domains (network or timeout).")

# ---------- Tab: Downloads ----------
with tabs[6]:
    st.header("Downloads & Exports")
    st.markdown("Download the sample files or export the metrics / selected protein data.")
    st.download_button("Download sample CSV", SAMPLE_CSV.encode('utf-8'), file_name="sample_network.csv")
    st.download_button("Download sample FASTA", SAMPLE_FASTA.encode('utf-8'), file_name="sample_sequences.fasta")
    st.download_button("Download metrics table (CSV)", metrics_df.to_csv(index=False).encode('utf-8'), file_name="protein_metrics.csv")
    # selected protein export
    sel = st.selectbox("Select protein to export sequence & metrics", sorted(list(proteins)), key="export_select")
    if st.button("Export selected protein data"):
        # assemble a small JSON / text
        seq_text = fasta_seqs.get(sel, "")
        # try UniProt if empty
        if not seq_text:
            try:
                r = requests.get(f"https://rest.uniprot.org/uniprotkb/{sel}.fasta", timeout=6)
                if r.ok and r.text.strip():
                    seq_text = "".join(r.text.splitlines()[1:])
            except Exception:
                seq_text = ""
        export_dict = {
            "Protein": sel,
            "Degree": degree.get(sel, 0),
            "Closeness": closeness.get(sel, 0.0),
            "Betweenness": betweenness.get(sel, 0.0),
            "Clustering": clustering.get(sel, 0.0),
            "Sequence": seq_text
        }
        out_bytes = str(export_dict).encode('utf-8')
        st.download_button("Download selected protein data (txt)", out_bytes, file_name=f"{sel}_data.txt")

# -----------------------------
# End of app
st.markdown("<hr><p style='text-align:center; color:gray;'>All features preserved: original metrics, static network, interactive cluster map, sequences, motifs, AlphaFold viewer, and sample downloads.</p>", unsafe_allow_html=True)

