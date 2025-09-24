# ipf_enhanced_app.py
# Keep requirements.txt unchanged: uses only streamlit, pandas, requests, py3Dmol, matplotlib
import streamlit as st
import pandas as pd
import requests
import py3Dmol
import matplotlib.pyplot as plt
import io
import base64
import math
from typing import List, Tuple, Dict, Optional
from collections import deque, defaultdict

# -------------------------------
# Config / Helpers
# -------------------------------
st.set_page_config(page_title="IPF Structure & Network Lab", layout="wide")
st.title("🫁 IPF Protein Structure & Network Explorer — Enhanced")

# Small CSS to tighten UI
st.markdown("""
<style>
.reportview-container .main .block-container{padding-top:0rem;}
.small-note {font-size:0.9rem; color:#444;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Caching remote requests
# -------------------------------
@st.cache_data(show_spinner=False)
def fetch_text(url: str, timeout: int = 10) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout)
        if r.ok:
            return r.text
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def uniprot_lookup_by_gene(gene_symbol: str) -> Dict:
    """
    Best-effort UniProt query for a human gene symbol (returns first hit).
    Uses UniProt REST API (public).
    """
    try:
        query = f"gene_exact:{gene_symbol} AND organism_id:9606"
        url = f"https://rest.uniprot.org/uniprotkb/search?query={requests.utils.quote(query)}&format=json&limit=1"
        r = requests.get(url, timeout=8)
        if r.ok:
            data = r.json()
            if data.get('results'):
                entry = data['results'][0]
                accession = entry.get('primaryAccession')
                protein_name = entry.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
                return {'accession': accession, 'protein_name': protein_name}
    except Exception:
        pass
    return {}

# -------------------------------
# Structure rendering (py3Dmol)
# -------------------------------
def render_structure(uniprot_id: str, pdb_id: Optional[str] = None, variants: Optional[List[str]] = None, highlight_neighbors: Optional[List[int]] = None):
    st.subheader(f"Protein Structure Viewer: {uniprot_id}")
    pdb_data = None
    if pdb_id:
        pdb_url = f"https://files.rcsb.org/view/{pdb_id}.pdb"
        pdb_data = fetch_text(pdb_url)
        if not pdb_data:
            st.warning("Could not fetch PDB file, falling back to AlphaFold.")
            pdb_id = None

    if not pdb_data:
        af_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        pdb_data = fetch_text(af_url)
        if not pdb_data:
            st.error("No structure available (PDB or AlphaFold).")
            return

    # 3D viewer
    try:
        viewer = py3Dmol.view(width=760, height=520)
        viewer.addModel(pdb_data, "pdb")
        viewer.setStyle({"cartoon": {"color": "spectrum"}})

        # highlight variant residues
        if variants:
            for v in variants:
                try:
                    ires = int(v)
                    viewer.addStyle({"resi": ires}, {"stick": {"colorscheme": "redCarbon"}})
                except:
                    pass

        # highlight HETATM ligands
        for line in pdb_data.splitlines():
            if line.startswith("HETATM"):
                try:
                    resi = int(line[22:26].strip())
                    viewer.addStyle({"resi": resi}, {"stick": {"color": "green"}})
                except:
                    pass

        # highlight neighbor residues (list of ints)
        if highlight_neighbors:
            for resi in highlight_neighbors:
                try:
                    viewer.addStyle({"resi": int(resi)}, {"sphere": {"color": "orange", "radius": 0.7}})
                except:
                    pass

        viewer.zoomTo()
        st.components.v1.html(viewer._make_html(), height=560)
    except Exception as e:
        st.error(f"3D viewer failed: {e}")

    # External link
    if pdb_id:
        st.markdown(f"🔗 [Open in RCSB PDB Viewer](https://www.rcsb.org/structure/{pdb_id})")
    else:
        st.markdown(f"🔗 [Open in AlphaFold Viewer](https://www.alphafold.ebi.ac.uk/entry/{uniprot_id})")

    # pLDDT plotting
    plDDT_scores = []
    for line in pdb_data.splitlines():
        if line.startswith("ATOM"):
            try:
                score = float(line[60:66].strip())
                plDDT_scores.append(score)
            except:
                pass

    if plDDT_scores:
        st.subheader("Intrinsic Disorder / Stability (per-residue pLDDT)")
        fig, ax = plt.subplots(figsize=(9,2.2))
        ax.plot(range(1, len(plDDT_scores)+1), plDDT_scores, color="purple", lw=1)
        ax.axhline(50, color="red", linestyle="--", label="pLDDT=50 (low confidence/disorder)")
        ax.set_xlabel("Residue Index")
        ax.set_ylabel("pLDDT")
        ax.set_ylim(0,100)
        ax.legend(loc='upper right', fontsize=8)
        st.pyplot(fig)

# -------------------------------
# Network building & analytics (no networkx)
# -------------------------------
def build_tripartite_edges(df: pd.DataFrame) -> Tuple[List[Tuple[str,str,str]], Dict[str,int]]:
    required = ['miRNA','Gene','Protein']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV must include column: {c}")
    df = df.copy()
    df['miRNA'] = df['miRNA'].astype(str).str.strip()
    df['Gene'] = df['Gene'].astype(str).str.strip()
    df['Protein'] = df['Protein'].astype(str).str.strip()
    edges = []
    nodes = set()
    for _, row in df.iterrows():
        mi = row['miRNA']
        gene = row['Gene']
        prot = row['Protein']
        if mi and gene:
            edges.append((mi, gene, 'miRNA->Gene'))
            nodes.add(mi); nodes.add(gene)
        if gene and prot:
            edges.append((gene, prot, 'Gene->Protein'))
            nodes.add(gene); nodes.add(prot)
    degree = {n:0 for n in nodes}
    for a,b,_ in edges:
        degree[a] = degree.get(a,0) + 1
        degree[b] = degree.get(b,0) + 1
    return edges, degree

def connected_components(edge_list: List[Tuple[str,str,str]]) -> List[set]:
    # undirected adjacency
    adj = defaultdict(set)
    for a,b,_ in edge_list:
        adj[a].add(b)
        adj[b].add(a)
    visited = set()
    comps = []
    for node in adj:
        if node not in visited:
            q = deque([node]); comp = set([node]); visited.add(node)
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v); q.append(v); comp.add(v)
            comps.append(comp)
    return comps

def betweenness_centrality(edge_list: List[Tuple[str,str,str]]) -> Dict[str,float]:
    # naive algorithm: Brandes-like but implemented without networkx
    # build adjacency
    adj = defaultdict(set)
    for a,b,_ in edge_list:
        adj[a].add(b); adj[b].add(a)
    nodes = list(adj.keys())
    CB = {v:0.0 for v in nodes}
    for s in nodes:
        # single-source shortest paths
        S = []
        P = {v:[] for v in nodes}
        sigma = dict.fromkeys(nodes, 0.0)
        dist = dict.fromkeys(nodes, -1)
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
        # accumulation
        delta = dict.fromkeys(nodes, 0.0)
        for w in reversed(S):
            for v in P[w]:
                delta_v = (sigma[v]/sigma[w]) * (1 + delta[w])
                delta[v] += delta_v
            if w != s:
                CB[w] += delta[w]
    # normalize for undirected graph: divide by 2 for pairs
    for k in CB:
        CB[k] = CB[k] / 2.0
    return CB

# -------------------------------
# Plotting utilities
# -------------------------------
def circular_positions(nodes: List[str]) -> Dict[str, Tuple[float,float]]:
    n = len(nodes)
    pos = {}
    for i,node in enumerate(nodes):
        angle = 2*math.pi*i/n
        pos[node] = (math.cos(angle), math.sin(angle))
    return pos

def plot_network(edge_list: List[Tuple[str,str,str]], degree: Dict[str,int], highlight_nodes: Optional[List[str]] = None, title: str = "miRNA–Gene–Protein Network"):
    nodes = sorted(list({n for e in edge_list for n in (e[0], e[1])}))
    if not nodes:
        fig, ax = plt.subplots()
        ax.text(0.5,0.5,"No network to display", ha='center')
        return fig
    pos = circular_positions(nodes)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.axis('off')
    max_deg = max(degree.values()) if degree else 1
    for a,b,etype in edge_list:
        x1,y1 = pos[a]; x2,y2 = pos[b]
        color = 'tab:blue' if 'miRNA' in etype else 'tab:green'
        ax.plot([x1,x2],[y1,y2], color=color, alpha=0.6, linewidth=0.8)
    for node in nodes:
        x,y = pos[node]
        deg = degree.get(node,0)
        size = 120*(0.5 + deg/max_deg)
        if node.lower().startswith('hsa') or node.lower().startswith('mir'):
            face = 'lightblue'; edge_col='navy'
        elif node.isupper() and len(node) <= 10:
            face = 'lightgreen'; edge_col='darkgreen'
        else:
            face='lightgray'; edge_col='gray'
        z = 3
        if highlight_nodes and node in highlight_nodes:
            ax.scatter([x],[y], s=size*1.6, facecolor='orange', edgecolors='red', zorder=z)
        else:
            ax.scatter([x],[y], s=size, facecolor=face, edgecolors=edge_col, zorder=z)
        ax.text(x,y,node, fontsize=7, ha='center', va='center', zorder=4)
    ax.set_title(title)
    return fig, pos

# -------------------------------
# Sample dataset generator
# -------------------------------
SAMPLE_CSV = """miRNA,Gene,Protein
hsa-miR-21,TGFB1,TGFB1
hsa-miR-29,COL1A1,COL1A1
hsa-miR-200c,ZEB1,ZEB1
hsa-miR-155,MMP7,MMP7
hsa-miR-326,TERT,TERT
"""

def sample_csv_download_button():
    b = SAMPLE_CSV.encode('utf-8')
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="ipf_sample.csv">⬇️ Download example CSV (IPF demo)</a>'
    st.markdown(href, unsafe_allow_html=True)

# -------------------------------
# App UI: Modes
# -------------------------------
mode = st.sidebar.radio("Mode", ["Single Protein", "Batch CSV", "Network Explorer", "Demo & Tutorial", "About"])
st.sidebar.markdown("**Quick actions**")
if st.sidebar.button("Download sample CSV"):
    sample_csv_download_button()

st.sidebar.markdown("---")
st.sidebar.markdown("**Tips:** Upload CSV with columns `miRNA,Gene,Protein` (headers required). Use UniProt accession in Protein column for best AlphaFold linking.")

# -------------------------------
# Single Protein Mode
# -------------------------------
if mode == "Single Protein":
    st.header("🔍 Single Protein Explorer")
    col1, col2 = st.columns([3,1])
    with col1:
        query = st.text_input("Protein name or UniProt ID (e.g., TGFB1 or Q9HC84)", value="")
        var_in = st.text_input("Comma-separated variant residue numbers (optional)", value="")
    with col2:
        st.markdown("**Options**")
        pdb_input = st.text_input("Optional PDB ID (e.g., 1D5R)", value="")
        if st.button("Lookup UniProt metadata"):
            if query:
                meta = uniprot_lookup_by_gene(query.strip())
                if meta:
                    st.success(f"Found UniProt accession: {meta.get('accession')} — {meta.get('protein_name')}")
                else:
                    st.warning("No UniProt entry found for that gene symbol.")
    if st.button("View Structure"):
        if not query:
            st.error("Enter a protein gene symbol or UniProt accession.")
        else:
            variants = [v.strip() for v in var_in.split(",")] if var_in else None
            pid = pdb_input.strip() if pdb_input else None
            render_structure(query.strip().upper(), pdb_id=pid, variants=variants)

# -------------------------------
# Batch CSV Mode
# -------------------------------
elif mode == "Batch CSV":
    st.header("📂 Batch CSV: per-row structure exploration & validation")
    uploaded = st.file_uploader("Upload CSV with columns miRNA,Gene,Protein (or use demo CSV)", type="csv")
    if uploaded is None:
        st.info("You can download and edit the example CSV to test functionality (sidebar).")
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.markdown("**Preview (first 20 rows)**")
            st.dataframe(df.head(20))
            st.markdown("**Row actions:** Click to view structure for that row's protein.")
            for idx,row in df.iterrows():
                cols = st.columns([4,1])
                with cols[0]:
                    st.write(f"**{row.get('Protein','')}**  — Gene: {row.get('Gene','')}  — miRNA: {row.get('miRNA','')}")
                with cols[1]:
                    if st.button("View structure", key=f"view_{idx}"):
                        protein = str(row.get('Protein','')).upper()
                        # try UniProt lookup for accession
                        meta = uniprot_lookup_by_gene(str(row.get('Gene','')).upper())
                        pdb_lookup = {"PTEN":"1D5R","AKT1":"4EKL"}
                        pdb_id = pdb_lookup.get(protein, None)
                        if meta and 'accession' in meta and not protein.startswith("Q"):
                            # if user provided gene, ask to use accession
                            try_use = st.radio("Use UniProt accession from lookup?", ["Use provided Protein column", f"Use accession {meta['accession']}"], index=0, key=f"acc_choice_{idx}")
                            if try_use.startswith("Use accession"):
                                protein = meta['accession']
                        render_structure(protein, pdb_id)
            # allow download of original file
            csv_bytes = df.to_csv(index=False).encode('utf-8')
            b64 = base64.b64encode(csv_bytes).decode()
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="uploaded_ipf.csv">⬇️ Download uploaded CSV</a>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

# -------------------------------
# Network Explorer
# -------------------------------
elif mode == "Network Explorer":
    st.header("🕸 Network Explorer — build tripartite network from CSV")
    upload = st.file_uploader("Upload CSV (miRNA,Gene,Protein)", type="csv", key="netfile")
    if upload:
        try:
            df = pd.read_csv(upload)
            st.success("CSV loaded.")
            st.subheader("Input preview")
            st.dataframe(df.head(50))

            # build edges and degree
            edge_list, degree = build_tripartite_edges(df)
            st.subheader("Network statistics")
            comps = connected_components(edge_list)
            st.markdown(f"- Nodes: **{len(degree)}**")
            st.markdown(f"- Edges: **{len(edge_list)}**")
            st.markdown(f"- Connected components: **{len(comps)}** (largest size: {max(len(c) for c in comps)})")

            # compute betweenness centrality (may take time for large graphs)
            if st.button("Compute betweenness centrality (naive; may be slow)"):
                with st.spinner("Computing betweenness..."):
                    bc = betweenness_centrality(edge_list)
                    bcd = pd.DataFrame(list(bc.items()), columns=['Node','Betweenness']).sort_values('Betweenness', ascending=False)
                    st.subheader("Top betweenness nodes")
                    st.dataframe(bcd.head(20))
            # show degree table
            ddf = pd.DataFrame(list(degree.items()), columns=['Node','Degree']).sort_values('Degree', ascending=False)
            st.subheader("Top degree nodes")
            st.dataframe(ddf.head(30))

            # rendering network
            st.markdown("---")
            st.subheader("Interactive network plot")
            plot_title = st.text_input("Plot title", value="IPF miRNA–Gene–Protein Network")
            # allow choosing a node to highlight neighbors
            node_choice = st.selectbox("Select node to highlight (optional)", options=[None]+ddf['Node'].tolist(), index=0)
            highlight_neighbors = []
            if node_choice:
                # compute neighbors list
                neighbors = set()
                for a,b,_ in edge_list:
                    if a == node_choice:
                        neighbors.add(b)
                    if b == node_choice:
                        neighbors.add(a)
                highlight_nodes = [node_choice] + list(neighbors)
                fig,pos = plot_network(edge_list, degree, highlight_nodes, title=plot_title)
            else:
                fig,pos = plot_network(edge_list, degree, None, title=plot_title)
            st.pyplot(fig)

            # allow node->structure quick view (via selectbox)
            prot_nodes = [n for n in ddf['Node'].tolist() if n.isupper() and len(n)<=10]
            if prot_nodes:
                sel_prot = st.selectbox("Quick view a protein (top nodes)", options=[None]+prot_nodes)
                if sel_prot:
                    pdb_map = {"PTEN":"1D5R","AKT1":"4EKL"}
                    pdbid = pdb_map.get(sel_prot, None)
                    render_structure(sel_prot, pdb_id=pdbid, variants=None, highlight_neighbors=None)

            # exports
            st.markdown("---")
            edf = pd.DataFrame(edge_list, columns=['Source','Target','Type'])
            st.markdown("Download network files")
            csv_e = edf.to_csv(index=False).encode('utf-8')
            csv_d = ddf.to_csv(index=False).encode('utf-8')
            b64e = base64.b64encode(csv_e).decode()
            b64d = base64.b64encode(csv_d).decode()
            st.markdown(f'<a href="data:file/csv;base64,{b64e}" download="ipf_edge_list.csv">⬇️ Download edge list CSV</a>', unsafe_allow_html=True)
            st.markdown(f'<a href="data:file/csv;base64,{b64d}" download="ipf_node_degree.csv">⬇️ Download node degree CSV</a>', unsafe_allow_html=True)

            # export network plot PNG
            save_png = st.button("Export network plot as PNG")
            if save_png:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                b64png = base64.b64encode(buf.read()).decode()
                href = f'<a href="data:image/png;base64,{b64png}" download="ipf_network.png">⬇️ Download network PNG</a>'
                st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not build network: {e}")

# -------------------------------
# Demo / Tutorial
# -------------------------------
elif mode == "Demo & Tutorial":
    st.header("Demo & Tutorial")
    st.markdown("""
    This demo walks through the app:
    1. Download the example CSV (sidebar).
    2. Open *Network Explorer* and upload the CSV.
    3. View network statistics, compute betweenness, and render the network.
    4. Use *Batch CSV* to open rows and view protein structures.
    """)
    st.markdown("**Why this app is useful**")
    st.markdown("- Lightweight and reproducible (CSV-first).  \n- Integrates network prioritization with structural inspection (AlphaFold/PDB).  \n- Good for rapid hypothesis generation in IPF research.")
    st.markdown("---")
    st.markdown("**Example CSV**")
    st.code(SAMPLE_CSV, language='csv')

# -------------------------------
# About
# -------------------------------
else:
    st.header("About & Notes")
    st.markdown("""
    **IPF Protein Structure & Network Explorer** — enhanced edition.

    Key design principles:
    - Minimal dependencies for portability.
    - CSV-first workflow for reproducibility.
    - Rapid integration of network prioritization and structure inspection.

    Limitations & future improvements (if you allow package changes):
    - Add `networkx` for advanced layouts and analytics.
    - Add `plotly` or D3 for interactive network visualization.
    - Add UniProt/STRING detailed annotation caching and pathway enrichment.
    """)

# -------------------------------
# Footer / contact
# -------------------------------
st.markdown("---")
st.markdown("Built for research prototyping. Keep prototypes reproducible and cite appropriate databases (UniProt, AlphaFold, STRING).")
