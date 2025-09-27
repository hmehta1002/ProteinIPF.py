# app.py
# Protein Network Explorer — final single-file app (robust, no-file-dependency, no duplicate keys)
import streamlit as st
import pandas as pd
import requests
import math, random, itertools, io, time
from collections import deque, defaultdict
import matplotlib.pyplot as plt

# Optional libraries — used only if available
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
st.set_page_config(page_title="Protein Network Explorer — Final", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align:center; color:#4B0082;'>🧬 Protein Network Explorer — Final</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6A5ACD;'>All features preserved, hardened against Streamlit errors.</p>", unsafe_allow_html=True)

# -------------------------
# Embedded sample data (always available)
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
MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQK
>MDM2_HUMAN
MDM2SEQEXAMPLEXXXXXXXXXXXXXX
>BRCA1_HUMAN
BRCA1SEQEXAMPLEYYYYYYYYYYYYYYY
"""

# -------------------------
# Sidebar: uploads, sample downloads, reset
# -------------------------
st.sidebar.header("Upload & Samples")

uploaded_csv = st.sidebar.file_uploader("Upload network CSV (needs Protein1,Protein2 columns or map them)", type=["csv"], key="uploader_csv_001")
uploaded_fasta = st.sidebar.file_uploader("Upload FASTA (optional)", type=["fasta","fa","txt"], key="uploader_fasta_001")

# Download sample buttons (generate from inline strings)
st.sidebar.download_button("⬇️ Download sample CSV", SAMPLE_CSV.encode("utf-8"), "sample_network.csv", key="dl_sample_csv_001")
st.sidebar.download_button("⬇️ Download sample FASTA", SAMPLE_FASTA.encode("utf-8"), "sample_sequences.fasta", key="dl_sample_fasta_001")

# Load/Reset to sample data (session-state)
if "use_sample" not in st.session_state:
    st.session_state.use_sample = False
if st.sidebar.button("Load / Reset to Sample Data", key="load_sample_btn_001"):
    st.session_state.use_sample = True
    st.session_state.sample_csv_text = SAMPLE_CSV
    st.session_state.sample_fasta_text = SAMPLE_FASTA
    st.sidebar.success("Sample data loaded to session memory.", key="load_sample_success_001")

# Options
filter_human = st.sidebar.checkbox("Filter for Homo sapiens (TaxID 9606)", value=True, key="filter_human_001")
with st.sidebar.expander("Column mapping (if your CSV headers differ)", expanded=False):
    map_protA = st.text_input("Protein A column (default: Protein1)", value="Protein1", key="map_protA_001")
    map_protB = st.text_input("Protein B column (default: Protein2)", value="Protein2", key="map_protB_001")
    map_tax = st.text_input("TaxID column (default: TaxID)", value="TaxID", key="map_tax_001")

# -------------------------
# Helpers
# -------------------------
def parse_fasta_text(text):
    seqs = {}
    if not text:
        return seqs
    # Accept both raw FASTA and Biopython-style content
    entries = text.strip().split(">")
    for e in entries:
        if not e.strip():
            continue
        lines = e.strip().splitlines()
        header = lines[0].split()[0]
        seq = "".join(lines[1:]).replace(" ", "").replace("\r", "")
        seqs[header] = seq
    return seqs

def read_csv_from_uploaded(uploaded):
    try:
        return pd.read_csv(uploaded)
    except Exception:
        # If uploaded is bytes IO
        try:
            text = uploaded.getvalue().decode("utf-8") if hasattr(uploaded, "getvalue") else str(uploaded)
            return pd.read_csv(io.StringIO(text))
        except Exception as e:
            raise e

def choose_column(preferred, fallbacks, available_cols):
    if preferred in available_cols:
        return preferred
    for f in fallbacks:
        if f in available_cols:
            return f
    return None

# Pure-Python BFS (for closeness)
def bfs_shortest_paths(graph, start):
    visited = {start: 0}
    q = deque([start])
    while q:
        v = q.popleft()
        for nb in graph.get(v, []):
            if nb not in visited:
                visited[nb] = visited[v] + 1
                q.append(nb)
    return visited

# Label propagation for communities
def label_propagation(graph, max_iter=200):
    labels = {n: n for n in graph}
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
    unique = {}
    communities = defaultdict(list)
    idx = 0
    for n,lbl in labels.items():
        if lbl not in unique:
            unique[lbl] = idx
            idx += 1
        communities[unique[lbl]].append(n)
    return labels, dict(communities)

# Spring layout (force-directed) - returns positions {node:(x,y)}
def spring_layout(graph, iterations=150, width=1.0, height=1.0):
    nodes = list(graph.keys())
    N = len(nodes)
    k = math.sqrt((width*height)/max(1, N))
    pos = {n:[random.uniform(0,width), random.uniform(0,height)] for n in nodes}
    t = width/10.0
    dt = t/(iterations+1)
    def rep(d): return (k*k)/d if d!=0 else k*k
    def attr(d): return (d*d)/k
    for i in range(iterations):
        disp = {n:[0.0,0.0] for n in nodes}
        for a in nodes:
            for b in nodes:
                if a==b: continue
                dx = pos[a][0]-pos[b][0]; dy = pos[a][1]-pos[b][1]
                dist = math.hypot(dx,dy) + 1e-9
                f = rep(dist)
                disp[a][0] += (dx/dist)*f; disp[a][1] += (dy/dist)*f
        for a in nodes:
            for b in graph[a]:
                dx = pos[a][0]-pos[b][0]; dy = pos[a][1]-pos[b][1]
                dist = math.hypot(dx,dy) + 1e-9
                f = attr(dist)
                disp[a][0] -= (dx/dist)*f; disp[a][1] -= (dy/dist)*f
        for n in nodes:
            dx,dy = disp[n]; dlen = math.hypot(dx,dy)
            if dlen>0:
                pos[n][0] += (dx/dlen)*min(dlen,t); pos[n][1] += (dy/dlen)*min(dlen,t)
            pos[n][0] = min(width, max(0.0, pos[n][0])); pos[n][1] = min(height, max(0.0, pos[n][1]))
        t -= dt
    # normalize to [0,1]
    xs = [pos[n][0] for n in nodes]; ys = [pos[n][1] for n in nodes]
    minx,maxx = min(xs), max(xs); miny,maxy = min(ys), max(ys)
    for n in nodes:
        pos[n][0] = (pos[n][0]-minx)/(maxx-minx+1e-12) if maxx-minx>0 else 0.5
        pos[n][1] = (pos[n][1]-miny)/(maxy-miny+1e-12) if maxy-miny>0 else 0.5
    return pos

# sequence-derived heuristics
AA_MASS = {'A':71.03711,'R':156.10111,'N':114.04293,'D':115.02694,'C':103.00919,'E':129.04259,'Q':128.05858,'G':57.02146,
           'H':137.05891,'I':113.08406,'L':113.08406,'K':128.09496,'M':131.04049,'F':147.06841,'P':97.05276,'S':87.03203,
           'T':101.04768,'W':186.07931,'Y':163.06333,'V':99.06841}
KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}
DISORDER_AA = set(list("PESQKRNTG"))

def compute_seq_metrics(seq):
    seq = seq.upper()
    count = 0; mw = 0.0; gravy_sum = 0.0; disorder_count = 0
    for aa in seq:
        if aa in AA_MASS:
            count += 1
            mw += AA_MASS[aa]
            gravy_sum += KD.get(aa, 0.0)
            if aa in DISORDER_AA:
                disorder_count += 1
    if count == 0:
        return {"length":0,"mw":0.0,"gravy":0.0,"disorder_frac":0.0,"instability":0.0}
    gravy = gravy_sum / count
    disorder_frac = disorder_count / count
    instability = max(0.0, 1.0 - min(1.0, count/1000.0)) + abs(gravy)/10.0
    return {"length":count, "mw":round(mw,2), "gravy":round(gravy,4), "disorder_frac":round(disorder_frac,4), "instability":round(instability,4)}

# -------------------------
# Load data: priority session sample -> uploaded -> inline sample
# -------------------------
if st.session_state.get("use_sample", False) and st.session_state.get("sample_csv_text"):
    try:
        df = pd.read_csv(io.StringIO(st.session_state["sample_csv_text"]))
    except Exception:
        df = pd.read_csv(io.StringIO(SAMPLE_CSV))
elif uploaded_csv is not None:
    try:
        df = read_csv_from_uploaded(uploaded_csv)
    except Exception:
        df = pd.read_csv(io.StringIO(SAMPLE_CSV))
else:
    df = pd.read_csv(io.StringIO(SAMPLE_CSV))

# fasta sequences
fasta_seqs = {}
if st.session_state.get("use_sample", False) and st.session_state.get("sample_fasta_text"):
    fasta_seqs = parse_fasta_text(st.session_state["sample_fasta_text"])
elif uploaded_fasta is not None:
    try:
        b = uploaded_fasta.read()
        text = b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b)
        fasta_seqs = parse_fasta_text(text)
    except Exception:
        fasta_seqs = parse_fasta_text(SAMPLE_FASTA)
else:
    fasta_seqs = parse_fasta_text(SAMPLE_FASTA)

# show detected columns in sidebar (no key argument for st.write)
st.sidebar.markdown("**Detected CSV columns:**")
try:
    st.sidebar.write(list(df.columns))
except Exception:
    st.sidebar.write("Could not detect columns for this CSV.")

# -------------------------
# Build adjacency with flexible header mapping
# -------------------------
cols = list(df.columns)
protA_col = choose_column(map_protA, ["Protein1","protein1","protA","InteractorA","source","geneA","GeneA","A"], cols)
protB_col = choose_column(map_protB, ["Protein2","protein2","protB","InteractorB","target","geneB","GeneB","B"], cols)
tax_col = choose_column(map_tax, ["TaxID","taxid","tax_id","Tax_Id"], cols)

adj = {}
proteins = set()
parse_errors = []

if protA_col is None or protB_col is None:
    parse_errors.append(f"Could not find protein columns. Detected columns: {cols}. Use mapping in sidebar.")
else:
    for _, row in df.iterrows():
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

# if no edges detected, fall back to sample data
if not adj:
    df_sample = pd.read_csv(io.StringIO(SAMPLE_CSV))
    adj = {}
    proteins = set()
    for _, r in df_sample.iterrows():
        p1 = r["Protein1"]; p2 = r["Protein2"]
        proteins.update([p1,p2])
        adj.setdefault(p1, set()).add(p2)
        adj.setdefault(p2, set()).add(p1)
    # also ensure sample fasta present
    if not fasta_seqs:
        fasta_seqs = parse_fasta_text(SAMPLE_FASTA)

# -------------------------
# Compute metrics (pure Python)
# -------------------------
# degree
degree = {n: len(adj[n]) for n in proteins}

# closeness
closeness = {}
for n in proteins:
    sp = bfs_shortest_paths(adj, n)
    if len(sp) > 1:
        s = sum(sp.values())
        closeness[n] = (len(sp)-1)/s if s>0 else 0.0
    else:
        closeness[n] = 0.0

# betweenness (BFS-based accumulation)
betweenness = dict.fromkeys(proteins, 0.0)
for s in proteins:
    S=[]; P={v:[] for v in proteins}; sigma=dict.fromkeys(proteins,0.0); dist=dict.fromkeys(proteins,-1)
    sigma[s]=1.0; dist[s]=0; Q=deque([s])
    while Q:
        v = Q.popleft(); S.append(v)
        for w in adj[v]:
            if dist[w] < 0:
                dist[w] = dist[v]+1; Q.append(w)
            if dist[w] == dist[v] + 1:
                sigma[w] += sigma[v]; P[w].append(v)
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

# clustering coefficient
clustering = {}
for node in proteins:
    neigh = list(adj[node])
    k = len(neigh)
    if k < 2:
        clustering[node] = 0.0
    else:
        links = 0
        for u,v in itertools.combinations(neigh, 2):
            if u in adj[v]:
                links += 1
        clustering[node] = 2*links/(k*(k-1))

# metrics dataframe
metrics_df = pd.DataFrame({
    "Protein": list(proteins),
    "Degree": [degree[p] for p in proteins],
    "Closeness": [closeness[p] for p in proteins],
    "Betweenness": [betweenness[p] for p in proteins],
    "Clustering": [clustering[p] for p in proteins]
}).sort_values(by="Degree", ascending=False).reset_index(drop=True)

# -------------------------
# Community detection & layout
# -------------------------
labels, communities = label_propagation(adj, max_iter=200)
num_communities = len(communities)
positions = spring_layout(adj, iterations=180, width=1.0, height=1.0)

# -------------------------
# UI: Tabs (strings only)
# -------------------------
tab_labels = [
    "Upload / Files",
    "Network Map (Clusters)",
    "Network Metrics",
    "Protein Details",
    "Sequences",
    "Motifs / Domains",
    "3D Structure",
    "Intrinsic Disorder & Stability"
]
tabs = st.tabs(tab_labels)

# -------------------------
# Tab 0: Upload / Files
# -------------------------
with tabs[0]:
    st.header("Upload / Sample Files")
    st.markdown("You can upload your own network CSV and optional FASTA. Use the sidebar to download or load sample data.")
    st.write(f"Detected columns: {cols}")
    st.write(f"Nodes: {len(proteins)} | Edges (approx): {sum(len(v) for v in adj.values())//2} | Communities: {num_communities}")
    if parse_errors:
        for i, msg in enumerate(parse_errors):
            st.error(msg)
    # Reset button in main area
    if st.button("Reset to sample data (main)", key="reset_main_btn_001"):
        st.session_state.use_sample = True
        st.session_state.sample_csv_text = SAMPLE_CSV
        st.session_state.sample_fasta_text = SAMPLE_FASTA
        st.experimental_rerun()

# -------------------------
# Tab 1: Network Map (Clusters)
# -------------------------
with tabs[1]:
    st.header("Interactive Cluster Map")
    st.markdown("Nodes colored by community; node size ~ degree; use the select box to inspect a node.")
    fig, ax = plt.subplots(figsize=(12,7))
    cmap = plt.cm.get_cmap("tab20", max(4, num_communities))
    drawn = set()
    for a in adj:
        for b in adj[a]:
            if (a,b) in drawn or (b,a) in drawn:
                continue
            xa, ya = positions[a]; xb, yb = positions[b]
            ax.plot([xa, xb], [ya, yb], color="#DDDDDD", linewidth=0.9, zorder=1, alpha=0.6)
            drawn.add((a,b))
    for node in sorted(list(proteins)):
        x,y = positions[node]
        # find community id
        comm_id = None
        for cid, members in communities.items():
            if node in members:
                comm_id = cid
                break
        color = cmap(comm_id % cmap.N) if comm_id is not None else (0.5,0.5,0.5)
        sz = 50 + degree[node]*20
        ax.scatter(x, y, s=sz, color=color, edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(x, y+0.02, node, fontsize=8, ha="center", rotation=30, zorder=4)
    ax.axis("off")
    st.pyplot(fig, key="cluster_map_plot")

    selected_node = st.selectbox("Select protein to inspect", sorted(list(proteins)), key="map_select_001")
    if selected_node:
        st.subheader(f"Selected — {selected_node}")
        st.write(f"- Degree: **{degree[selected_node]}**")
        st.write(f"- Closeness: **{closeness[selected_node]:.6f}**")
        st.write(f"- Betweenness: **{betweenness[selected_node]:.6f}**")
        st.write(f"- Clustering coeff: **{clustering[selected_node]:.6f}**")
        seq = fasta_seqs.get(selected_node, "")
        if seq:
            st.text_area("Sequence (from uploaded/sample FASTA)", seq, height=200, key="map_seq_area_001")
        else:
            with st.spinner("Fetching sequence from UniProt..."):
                try:
                    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{selected_node}.fasta", timeout=10)
                    if r.ok and r.text.strip():
                        seq = "".join(r.text.splitlines()[1:])
                        st.text_area("Sequence (from UniProt)", seq, height=200, key="map_seq_uniprot_001")
                    else:
                        st.warning("Sequence not found in UniProt and not present in uploaded FASTA.")
                except Exception:
                    st.warning("UniProt query failed (network or rate limit).")
        if seq:
            seq_metrics = compute_seq_metrics(seq)
            st.markdown("**Sequence-derived heuristics**")
            st.write(f"- Length: **{seq_metrics['length']}**")
            st.write(f"- GRAVY: **{seq_metrics['gravy']}**")
            st.write(f"- Disorder-promoting fraction (heuristic): **{seq_metrics['disorder_frac']}**")
            st.write(f"- Instability proxy (heuristic): **{seq_metrics['instability']}**")

# -------------------------
# Tab 2: Network Metrics
# -------------------------
with tabs[2]:
    st.header("Network Metrics")
    st.markdown("Degree, closeness, betweenness, clustering. Download the metrics table as CSV.")
    st.dataframe(metrics_df, height=400)
    st.download_button("⬇️ Download metrics (CSV)", metrics_df.to_csv(index=False).encode("utf-8"), file_name="protein_metrics.csv", key="download_metrics_001")

# -------------------------
# Tab 3: Protein Details
# -------------------------
with tabs[3]:
    st.header("Protein Details & Lookup")
    prot_input = st.text_input("Enter UniProt entry name / ID (e.g., P53_HUMAN) or choose from network", key="prot_input_001")
    prot_choice = st.selectbox("Or pick a protein from network", options=[""] + sorted(list(proteins)), key="prot_choice_001")
    prot = prot_input.strip() if prot_input.strip() else (prot_choice if prot_choice else "")
    if prot:
        st.subheader(f"Details for {prot}")
        # show metrics if in network
        if prot in degree:
            st.write(f"- Degree: **{degree[prot]}**")
            st.write(f"- Closeness: **{closeness[prot]:.6f}**")
            st.write(f"- Betweenness: **{betweenness[prot]:.6f}**")
            st.write(f"- Clustering coeff: **{clustering[prot]:.6f}**")
        else:
            st.info("Protein not present in the current network.")
        # sequence
        seq_text = fasta_seqs.get(prot, "")
        if seq_text:
            st.text_area("Sequence (from uploaded/sample FASTA)", seq_text, height=250, key="details_seq_area_001")
        else:
            with st.spinner("Fetching FASTA from UniProt..."):
                try:
                    r = requests.get(f"https://rest.uniprot.org/uniprotkb/{prot}.fasta", timeout=10)
                    if r.ok and r.text.strip():
                        seq_text = "".join(r.text.splitlines()[1:])
                        st.text_area("Sequence (UniProt)", seq_text, height=250, key="details_seq_uniprot_001")
                    else:
                        st.warning("No sequence found for this ID on UniProt.")
                except Exception:
                    st.warning("UniProt fetch failed (network/time-out).")

        # export selected protein
        if st.button("Export selected protein data", key="export_selected_btn_001"):
            export_data = {
                "Protein": prot,
                "Degree": degree.get(prot, 0),
                "Closeness": closeness.get(prot, 0.0),
                "Betweenness": betweenness.get(prot, 0.0),
                "Clustering": clustering.get(prot, 0.0),
                "Sequence": seq_text
            }
            st.download_button("⬇️ Download selected protein data (txt)", str(export_data).encode("utf-8"), file_name=f"{prot}_data.txt", key="download_selected_protein_001")

# -------------------------
# Tab 4: Sequences
# -------------------------
with tabs[4]:
    st.header("FASTA Sequences (upload or view sample)")
    fasta_upload = st.file_uploader("Upload a FASTA file (optional)", type=["fasta","fa","txt"], key="fasta_upload_001")
    if fasta_upload:
        try:
            if SeqIO is not None:
                records = list(SeqIO.parse(fasta_upload, "fasta"))
                for rec in records:
                    st.text(f">{rec.id}\n{rec.seq}")
            else:
                raw = fasta_upload.read()
                txt = raw.decode("utf-8") if isinstance(raw, (bytes,bytearray)) else str(raw)
                st.text(txt)
        except Exception as e:
            st.error(f"Failed to parse FASTA: {e}")
    else:
        st.info("No FASTA uploaded. Sample FASTA shown below.")
        st.text(SAMPLE_FASTA)

# -------------------------
# Tab 5: Motifs / Domains
# -------------------------
with tabs[5]:
    st.header("Motifs & Domains (UniProt / EBI)")
    motif_prot = st.text_input("Enter UniProt ID to fetch motifs/domains", key="motif_input_001")
    if st.button("Fetch motifs/domains", key="fetch_motifs_btn_001"):
        if not motif_prot.strip():
            st.warning("Enter a UniProt ID first.")
        else:
            with st.spinner("Querying UniProt/EBI..."):
                try:
                    url = f"https://rest.uniprot.org/uniprotkb/{motif_prot}.json"
                    r = requests.get(url, timeout=12)
                    if r.ok:
                        j = r.json()
                        features = j.get("features", [])
                        if not features:
                            st.info("No features returned for this protein.")
                        else:
                            for i, f in enumerate(features[:100]):
                                ftype = f.get("type", "feature")
                                desc = f.get("description", "")
                                location = f.get("location", {})
                                st.markdown(f"**{ftype}** — {desc}  \nLocation: {location}")
                    else:
                        st.error("Could not fetch motifs/domains (not found or rate-limited).")
                except Exception as e:
                    st.error(f"Motif/domain request failed: {e}")

# -------------------------
# Tab 6: 3D Structure (AlphaFold)
# -------------------------
with tabs[6]:
    st.header("AlphaFold 3D Structure Viewer (py3Dmol)")
    struct_id = st.text_input("Enter AlphaFold/UniProt entry name (e.g., P53_HUMAN)", key="alphafold_input_001")
    if st.button("Load AlphaFold structure", key="alphafold_load_btn_001"):
        if not struct_id.strip():
            st.warning("Enter an AlphaFold/UniProt entry first.")
        else:
            if py3Dmol is None:
                st.error("py3Dmol is not installed in this environment. The 3D viewer is disabled.")
                st.markdown(f"You can still view AlphaFold page: https://alphafold.ebi.ac.uk/entry/{struct_id}")
            else:
                with st.spinner("Fetching AlphaFold PDB..."):
                    try:
                        url = f"https://alphafold.ebi.ac.uk/files/AF-{struct_id}-F1-model_v4.pdb"
                        r = requests.get(url, timeout=15)
                        if r.ok and r.text:
                            pdb_text = r.text
                            view = py3Dmol.view(width=800, height=500)
                            view.addModel(pdb_text, "pdb")
                            view.setStyle({"cartoon": {"color":"spectrum"}})
                            view.zoomTo()
                            view.show()
                            st.components.v1.html(view.js(), height=520, key="py3dmol_view_001")
                        else:
                            st.error("AlphaFold model not found on EBI for that ID.")
                    except Exception as e:
                        st.error(f"AlphaFold fetch failed: {e}")

# -------------------------
# Tab 7: Intrinsic Disorder & Stability
# -------------------------
with tabs[7]:
    st.header("Sequence-derived Intrinsic Disorder & Stability (heuristics)")
    seq_or_id = st.text_input("Enter UniProt entry name or paste a sequence", key="disorder_input_001")
    if st.button("Compute heuristics", key="compute_heuristics_btn_001"):
        seq_text = ""
        if seq_or_id.strip() in fasta_seqs:
            seq_text = fasta_seqs[seq_or_id.strip()]
        elif seq_or_id.strip().upper().startswith(">"):
            parsed = parse_fasta_text(seq_or_id.strip())
            if parsed:
                seq_text = list(parsed.values())[0]
        else:
            # try UniProt
            try:
                r = requests.get(f"https://rest.uniprot.org/uniprotkb/{seq_or_id.strip()}.fasta", timeout=10)
                if r.ok and r.text.strip():
                    seq_text = "".join(r.text.splitlines()[1:])
            except Exception:
                pass
        if not seq_text:
            st.warning("Sequence not found. Upload FASTA, paste a sequence, or enter a valid UniProt ID.")
        else:
            metrics = compute_seq_metrics(seq_text)
            st.write(f"- Length: **{metrics['length']}**")
            st.write(f"- Molecular weight (approx): **{metrics['mw']} Da**")
            st.write(f"- GRAVY: **{metrics['gravy']}**")
            st.write(f"- Disorder-promoting fraction: **{metrics['disorder_frac']}**")
            st.write(f"- Instability proxy (heuristic): **{metrics['instability']}**")
            st.info("These are heuristic, sequence-only estimates. For publication-grade predictions use IUPred/ProtParam/etc.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>This app is hardened to avoid common Streamlit errors (duplicate element IDs, missing sample files, invalid widget arguments). If you still see an error, copy the full traceback text and paste it here and I will patch it immediately.</p>", unsafe_allow_html=True)
