# app.py (FULL, integrated with AlphaFold download + NGL.js viewer)
import streamlit as st
import streamlit.components.v1 as components
import json
import csv
import os
import io
import math
import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import gzip
import base64
from io import BytesIO
from werkzeug.utils import secure_filename
from datetime import datetime

# -------------------------------
# Configuration
# -------------------------------
st.set_page_config(page_title="IPF-miNet Explorer (Research)", layout="wide", initial_sidebar_state="expanded")
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
EXPORT_DIR = "exports"
os.makedirs(EXPORT_DIR, exist_ok=True)
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

# -------------------------------
# Utility helpers
# -------------------------------
def safe_save_uploaded(uploaded_file, folder=UPLOADS_DIR):
    os.makedirs(folder, exist_ok=True)
    fname = secure_filename(uploaded_file.name)
    path = os.path.join(folder, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{fname}")
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    return path

# -------------------------------
# File parsing & validation
# -------------------------------
REQUIRED_COLUMNS = {"miRNA", "Gene", "Protein"}

def read_uploaded_file_from_bytes(content_bytes, filename):
    try:
        text = content_bytes.decode('utf-8')
    except Exception:
        text = content_bytes.decode('latin-1', errors='ignore')
    if filename.lower().endswith('.json'):
        data = json.loads(text)
        if isinstance(data, dict):
            for k in ("interactions", "data", "rows", "entries"):
                if k in data and isinstance(data[k], list):
                    return data[k], 'json'
            if all(isinstance(v, dict) for v in data.values()):
                return [ {"id": k, **v} for k, v in data.items() ], 'json'
            return [], 'json'
        elif isinstance(data, list):
            return data, 'json'
        else:
            return [], 'json'
    else:
        reader = csv.DictReader(io.StringIO(text))
        rows = [r for r in reader]
        return rows, 'csv'

def read_uploaded_file(uploaded_file):
    name = uploaded_file.name
    content = uploaded_file.read()
    return read_uploaded_file_from_bytes(content, name)

def validate_rows(rows):
    if not rows:
        return False, ["No rows found in uploaded file."], []
    cols = set(k.strip() for k in rows[0].keys())
    missing = REQUIRED_COLUMNS - {c for c in cols}
    if missing:
        return False, [f"Missing required columns: {', '.join(missing)}"], []
    normalized = []
    for r in rows:
        nr = {}
        for k, v in r.items():
            if v is None:
                v = ""
            nr[k.strip()] = str(v).strip()
        normalized.append(nr)
    return True, [], normalized

# -------------------------------
# Reproducible local DB (sample)
# -------------------------------
SAMPLE_MIRNA_TARGETS = [
    {"miRNA":"miR-21-5p","Gene":"PTEN","Evidence":"Validated","Source":"miRTarBase"},
    {"miRNA":"miR-29a","Gene":"COL1A1","Evidence":"Validated","Source":"miRTarBase"},
    {"miRNA":"miR-199a-5p","Gene":"TGFBR1","Evidence":"Predicted","Source":"miRDB"},
]
SAMPLE_PROTEIN_INTERACTIONS = [
    {"ProteinA":"PTEN","ProteinB":"AKT1","Score":"0.9","Source":"STRING"},
    {"ProteinA":"COL1A1","ProteinB":"COL1A2","Score":"0.95","Source":"STRING"},
    {"ProteinA":"TGFBR1","ProteinB":"SMAD3","Score":"0.88","Source":"STRING"},
]
SAMPLE_GENE_INFO = [
    {"Gene":"PTEN","Symbol":"PTEN","Function":"Tumor suppressor; negative regulator of PI3K/AKT"},
    {"Gene":"COL1A1","Symbol":"COL1A1","Function":"Collagen type I alpha 1 chain"},
    {"Gene":"TGFBR1","Symbol":"TGFBR1","Function":"TGF-beta receptor 1"},
]

def ensure_sample_db():
    mirna_fp = os.path.join(DATA_DIR, "mirna_targets.csv")
    prot_fp = os.path.join(DATA_DIR, "protein_interactions.csv")
    gene_fp = os.path.join(DATA_DIR, "gene_info.csv")
    if not os.path.exists(mirna_fp):
        pd.DataFrame(SAMPLE_MIRNA_TARGETS).to_csv(mirna_fp, index=False)
    if not os.path.exists(prot_fp):
        pd.DataFrame(SAMPLE_PROTEIN_INTERACTIONS).to_csv(prot_fp, index=False)
    if not os.path.exists(gene_fp):
        pd.DataFrame(SAMPLE_GENE_INFO).to_csv(gene_fp, index=False)

def load_local_db():
    ensure_sample_db()
    mirna_targets = pd.read_csv(os.path.join(DATA_DIR, "mirna_targets.csv")).to_dict(orient='records')
    protein_inter = pd.read_csv(os.path.join(DATA_DIR, "protein_interactions.csv")).to_dict(orient='records')
    gene_info = pd.read_csv(os.path.join(DATA_DIR, "gene_info.csv")).to_dict(orient='records')
    return mirna_targets, protein_inter, gene_info

# -------------------------------
# Normalization & indexing
# -------------------------------
def normalize_mirna_name(name):
    if not name:
        return ""
    return name.strip().lower().replace("mir", "mir").replace("-","").replace(" ", "")

def normalize_gene_symbol(symbol):
    if not symbol:
        return ""
    return symbol.strip().upper()

def build_index_maps(mirna_targets, protein_inter, gene_info):
    mirna_map = {}
    gene_map = {}
    protein_map = {}
    for rec in mirna_targets:
        key = normalize_mirna_name(rec.get("miRNA",""))
        mirna_map.setdefault(key, []).append(rec)
    for rec in protein_inter:
        a = normalize_gene_symbol(rec.get("ProteinA",""))
        b = normalize_gene_symbol(rec.get("ProteinB",""))
        protein_map.setdefault(a, []).append(rec)
        protein_map.setdefault(b, []).append(rec)
    for rec in gene_info:
        gene_map[ normalize_gene_symbol(rec.get("Gene","")) ] = rec
    return mirna_map, protein_map, gene_map

# -------------------------------
# Network construction & scoring (copied & slightly adjusted)
# -------------------------------
def build_network_from_selection(rows, mirna_index, protein_index, gene_index, scoring_params=None):
    edges = []
    nodes = {}

    for r in rows:
        mirna_raw = r.get("miRNA","")
        gene_raw = r.get("Gene","")
        prot_raw = r.get("Protein","")

        mirna_id = normalize_mirna_name(mirna_raw)
        gene_id = normalize_gene_symbol(gene_raw)
        prot_id = normalize_gene_symbol(prot_raw)

        nodes.setdefault(mirna_id, {"id":mirna_id, "label":mirna_raw, "type":"miRNA", "source":"uploaded"})
        nodes.setdefault(gene_id, {"id":gene_id, "label":gene_raw, "type":"Gene", "source":"uploaded"})
        nodes.setdefault(prot_id, {"id":prot_id, "label":prot_raw, "type":"Protein", "source":"uploaded"})

        edges.append({"source":mirna_id, "target":gene_id, "type":"miRNA-Gene", "score":1.0, "evidence":"uploaded"})
        edges.append({"source":gene_id, "target":prot_id, "type":"Gene-Protein", "score":1.0, "evidence":"uploaded"})

    for user_mirna in {normalize_mirna_name(r.get("miRNA","")) for r in rows}:
        for rec in mirna_index.get(user_mirna, []):
            tgt = normalize_gene_symbol(rec.get("Gene",""))
            nodes.setdefault(tgt, {"id":tgt, "label":tgt, "type":"Gene", "source":rec.get("Source","db")})
            nodes.setdefault(user_mirna, {"id":user_mirna, "label":user_mirna, "type":"miRNA", "source":"uploaded"})
            score = 0.9 if str(rec.get("Evidence","")).lower().startswith("valid") else 0.6
            edges.append({"source":user_mirna, "target":tgt, "type":"miRNA-Gene", "score":score, "evidence":rec.get("Source","db")})

    present_proteins = [n for n,info in nodes.items() if info['type'] in ('Protein','Gene')]
    for prot in present_proteins:
        for rec in protein_index.get(prot, []):
            a = normalize_gene_symbol(rec.get("ProteinA",""))
            b = normalize_gene_symbol(rec.get("ProteinB",""))
            try:
                score = float(rec.get("Score", 0.5))
            except Exception:
                score = 0.5
            nodes.setdefault(a, {"id":a, "label":a, "type":"Protein", "source":rec.get("Source","db")})
            nodes.setdefault(b, {"id":b, "label":b, "type":"Protein", "source":rec.get("Source","db")})
            edges.append({"source":a, "target":b, "type":"Protein-Protein", "score":score, "evidence":rec.get("Source","db")})

    # deduplicate edges
    edge_map = {}
    for e in edges:
        key = tuple(sorted([e['source'], e['target']]) + [e['type']])
        if key in edge_map:
            edge_map[key]['score'] = max(edge_map[key]['score'], e['score'])
            if e['evidence'] not in edge_map[key]['evidence']:
                edge_map[key]['evidence'] += f";{e['evidence']}"
        else:
            edge_map[key] = e.copy()
    final_edges = list(edge_map.values())
    final_nodes = list(nodes.values())

    for e in final_edges:
        if e.get('evidence','').lower() == 'uploaded':
            e['novelty'] = 1.0
        else:
            e['novelty'] = round(1.0 - float(min(max(e.get('score',0.0), 0.0), 1.0)), 3)

    return {"nodes": final_nodes, "edges": final_edges, "meta": {"node_count":len(final_nodes), "edge_count":len(final_edges)}}

# -------------------------------
# Plotly network visualization
# -------------------------------
def create_plotly_network(network_data):
    if not network_data or not network_data['nodes']:
        return go.Figure()

    G = nx.Graph()
    for n in network_data['nodes']:
        G.add_node(n['id'], **n)
    for e in network_data['edges']:
        G.add_edge(e['source'], e['target'], **e)

    pos = nx.spring_layout(G, seed=42, k=0.6)

    edge_x, edge_y, edge_text = [], [], []
    for u, v, edata in G.edges(data=True):
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        edge_text.append(f"{u}—{v} | {edata.get('type')} | s={edata.get('score')} | nov={edata.get('novelty')}")

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='text', text=edge_text, mode='lines')

    node_traces = []
    type_colors = {"miRNA":"#FF6B6B","Gene":"#4ECDC4","Protein":"#45B7D1"}
    for ntype in sorted({n.get('type') for n in network_data['nodes'] if n.get('type')}):
        xs, ys, texts, sizes = [], [], [], []
        for n in network_data['nodes']:
            if n.get('type') != ntype:
                continue
            x, y = pos[n['id']]
            xs.append(x); ys.append(y)
            texts.append(f"{n.get('label')} ({n.get('id')}) | {n.get('type')} | {n.get('source')}")
            sizes.append(20 if n.get('source')=='uploaded' else 12)
        trace = go.Scatter(x=xs, y=ys, mode='markers+text', text=[t.split(" | ")[0] for t in texts],
                           textposition='top center', hovertext=texts, hoverinfo='text',
                           marker=dict(size=sizes, color=type_colors.get(ntype,"#888"), line=dict(width=1, color='white')),
                           name=ntype)
        node_traces.append(trace)

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(title="IPF miRNA-Gene-Protein Network", showlegend=True, margin=dict(l=20,r=20,t=40,b=20))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig

# -------------------------------
# UniProt & RCSB helpers
# -------------------------------
UNIPROT_BASE = "https://rest.uniprot.org"
RCSB_BASE = "https://search.rcsb.org/rcsbsearch/v2/query"

def query_uniprot_by_gene(gene_symbol, organism="Homo sapiens"):
    query = f'gene_exact:{gene_symbol} AND organism_name:"{organism}"'
    url = f"{UNIPROT_BASE}/uniprot/search?query={requests.utils.quote(query)}&format=json&size=10"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data.get("results"):
            entry = data["results"][0]
            accession = entry.get("primaryAccession")
            prot_name = entry.get("proteinDescription",{}).get("recommendedName",{}).get("fullName",{}).get("value","")
            length = entry.get("sequence",{}).get("length")
            return {"accession": accession, "name": prot_name, "length": length, "raw": entry}
        else:
            return {}
    except Exception as e:
        return {"error": str(e)}

def query_pdb_by_uniprot(uniprot_acc):
    try:
        payload = {
          "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
              "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
              "operator": "exact_match",
              "value": uniprot_acc
            }
          },
          "request_options": {"return_all_hits": True},
          "return_type": "entry"
        }
        r = requests.post(RCSB_BASE, json=payload, timeout=10)
        r.raise_for_status()
        res = r.json()
        pdb_ids = [item['identifier'] for item in res.get('result_set', [])]
        return pdb_ids
    except Exception as e:
        return {"error": str(e)}

def alphafold_link_for_uniprot(uniprot_acc):
    return f"https://alphafold.ebi.ac.uk/entry/{uniprot_acc}"

# -------------------------------
# AlphaFold download + NGL embed
# -------------------------------
def download_alphafold_pdb(uniprot_acc, save_dir=EXPORT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    base_urls = [
        f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_acc}-F1-model_v2.pdb",
        f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_acc}-F1-model_v2.pdb.gz",
        f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_acc}-F1-model_v1.pdb",
        f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_acc}-F1-model_v1.pdb.gz",
    ]
    headers = {"User-Agent": "IPF-miNet-Explorer/1.0"}
    for url in base_urls:
        try:
            r = requests.get(url, timeout=15, headers=headers)
            if r.status_code == 200 and r.content:
                if url.endswith(".gz") or (r.content[:2] == b'\x1f\x8b'):
                    try:
                        bio = BytesIO(r.content)
                        with gzip.GzipFile(fileobj=bio) as gz:
                            pdb_bytes = gz.read()
                    except Exception:
                        try:
                            import zlib
                            pdb_bytes = zlib.decompress(r.content, 16+zlib.MAX_WBITS)
                        except Exception:
                            continue
                else:
                    pdb_bytes = r.content
                try:
                    pdb_text = pdb_bytes.decode('utf-8')
                except Exception:
                    pdb_text = pdb_bytes.decode('latin-1', errors='ignore')
                safe_name = f"AF-{uniprot_acc}-model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdb"
                local_path = os.path.join(save_dir, safe_name)
                with open(local_path, "w", encoding='utf-8') as fw:
                    fw.write(pdb_text)
                return local_path, pdb_text
            else:
                continue
        except Exception:
            continue
    return None, None

def embed_pdb_with_ngl(pdb_text, element_id="nglview", width="100%", height="520px"):
    b64 = base64.b64encode(pdb_text.encode('utf-8')).decode('ascii')
    html = f"""
    <div id="{element_id}" style="width:{width}; height:{height};"></div>
    <script src="https://unpkg.com/ngl@0.10.4/dist/ngl.js"></script>
    <script>
    (function(){{
        const stage = new NGL.Stage('{element_id}');
        const b64 = "{b64}";
        const bin = atob(b64);
        const len = bin.length;
        const arr = new Uint8Array(len);
        for (let i = 0; i < len; i++) {{
            arr[i] = bin.charCodeAt(i);
        }}
        const blob = new Blob([arr], {{type: 'text/plain'}});
        const url = URL.createObjectURL(blob);
        stage.loadFile(url, {{ defaultRepresentation: true }}).then(function(comp) {{
            stage.autoView();
            comp.addRepresentation("cartoon", {{color: "chainname"}});
            comp.addRepresentation("surface", {{opacity:0.2}});
        }});
        window.addEventListener('resize', function() {{ stage.handleResize(); }});
    }})();
    </script>
    """
    return html

# -------------------------------
# Streamlit UI (main)
# -------------------------------
def main():
    st.title("🧬 IPF-miNet Explorer — Research Pipeline (AlphaFold + NGL viewer)")

    left, right = st.columns([2,1])

    with left:
        st.header("Upload interaction table (CSV/JSON)")
        uploaded_file = st.file_uploader("Upload interactions (CSV or JSON)", type=["csv","json"])
        use_db = st.checkbox("Use local curated DB expansion", value=True)
        show_tables = st.checkbox("Show parsed tables", value=True)

        if uploaded_file:
            saved_path = safe_save_uploaded(uploaded_file)
            with open(saved_path, "rb") as fh:
                rows, ftype = read_uploaded_file_from_bytes(fh.read(), os.path.basename(saved_path))
            valid, errors, norm_rows = validate_rows(rows)
            if not valid:
                st.error("Upload validation failed:")
                for e in errors:
                    st.write("- " + e)
            else:
                st.success(f"Uploaded parsed as {ftype.upper()} with {len(norm_rows)} rows")
                if show_tables:
                    st.subheader("Preview (first 20 rows)")
                    st.dataframe(pd.DataFrame(norm_rows).head(20))

                mirna_targets, protein_inter, gene_info = load_local_db()
                mirna_index, protein_index, gene_index = build_index_maps(mirna_targets, protein_inter, gene_info)
                with st.spinner("Building network and expanding using local DB..."):
                    network_data = build_network_from_selection(norm_rows, mirna_index, protein_index, gene_index)
                st.session_state['network_data'] = network_data

                st.subheader("Network summary")
                cols = st.columns(4)
                cols[0].metric("Nodes", network_data['meta']['node_count'])
                cols[1].metric("Edges", network_data['meta']['edge_count'])
                degs = {}
                for e in network_data['edges']:
                    degs[e['source']] = degs.get(e['source'],0) + 1
                    degs[e['target']] = degs.get(e['target'],0) + 1
                deg_df = pd.DataFrame([{"Node":k,"Degree":v} for k,v in degs.items()]).sort_values("Degree", ascending=False).head(10)
                cols[2].metric("Top degree", f"{deg_df.iloc[0]['Node']} ({int(deg_df.iloc[0]['Degree'])})" if not deg_df.empty else "N/A")
                cols[3].metric("Novel candidate edges", sum(1 for e in network_data['edges'] if e.get('novelty',0) > 0.5))

                st.plotly_chart(create_plotly_network(network_data), use_container_width=True, height=600)

                # exports
                st.subheader("Export / Download")
                if st.button("Download nodes CSV"):
                    nodes_df = pd.DataFrame(network_data['nodes'])
                    path = os.path.join(EXPORT_DIR, f"nodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    nodes_df.to_csv(path, index=False)
                    with open(path, "rb") as f:
                        st.download_button("Download nodes CSV file", f, file_name=os.path.basename(path))
                if st.button("Download edges CSV"):
                    edges_df = pd.DataFrame(network_data['edges'])
                    path = os.path.join(EXPORT_DIR, f"edges_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                    edges_df.to_csv(path, index=False)
                    with open(path, "rb") as f:
                        st.download_button("Download edges CSV file", f, file_name=os.path.basename(path))
                if st.button("Download graph (GraphML)"):
                    G = nx.Graph()
                    for n in network_data['nodes']:
                        G.add_node(n['id'], **n)
                    for e in network_data['edges']:
                        G.add_edge(e['source'], e['target'], **e)
                    gpath = os.path.join(EXPORT_DIR, f"network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.graphml")
                    nx.write_graphml(G, gpath)
                    with open(gpath, "rb") as f:
                        st.download_button("Download GraphML", f, file_name=os.path.basename(gpath))

    with right:
        st.header("Protein / Structure Lookup")
        gene_query = st.text_input("Enter gene symbol (e.g., LOXL2)", key="protein_lookup")
        if gene_query:
            st.info("Querying UniProt & RCSB for structural availability...")
            uni = query_uniprot_by_gene(gene_query, organism="Homo sapiens")
            if not uni:
                st.warning("No UniProt entry found. Check gene symbol or species.")
            elif "error" in uni:
                st.error(f"UniProt query error: {uni['error']}")
            else:
                st.subheader(f"{gene_query.upper()} — UniProt")
                st.write(f"**Accession:** {uni.get('accession','N/A')}")
                st.write(f"**Protein name:** {uni.get('name','N/A')}")
                st.write(f"**Length:** {uni.get('length','N/A')}")
                acc = uni.get('accession')
                if acc:
                    pdbs = query_pdb_by_uniprot(acc)
                    if isinstance(pdbs, dict) and 'error' in pdbs:
                        st.error("RCSB query error: " + pdbs['error'])
                    else:
                        if pdbs:
                            st.success(f"Found {len(pdbs)} PDB entries.")
                            st.write(", ".join(pdbs))
                            for pid in pdbs[:10]:
                                st.markdown(f"- [{pid}](https://www.rcsb.org/structure/{pid})")
                        else:
                            st.warning("No PDB entries found for this UniProt accession.")
                            st.info("Attempting to download AlphaFold model for this UniProt accession...")
                            af_path, pdb_text = download_alphafold_pdb(acc)
                            if af_path and pdb_text:
                                st.success(f"AlphaFold model downloaded: {os.path.basename(af_path)}")
                                ngl_html = embed_pdb_with_ngl(pdb_text, element_id=f"ngl_{acc}")
                                components.html(ngl_html, height=560)
                                with open(af_path, "rb") as f:
                                    st.download_button("Download AlphaFold PDB", data=f, file_name=os.path.basename(af_path))
                            else:
                                st.error("Could not find an AlphaFold PDB file automatically.")
                                st.markdown(f"[AlphaFold page]({alphafold_link_for_uniprot(acc)})")
                else:
                    st.info("No accession available from UniProt result.")

        st.markdown("---")
        st.subheader("Example protein: LOXL2")
        st.write("LOXL2 — ECM crosslinking enzyme implicated in IPF. Try searching 'LOXL2' above.")
        st.markdown("- PDB example: [5ZE3](https://www.rcsb.org/structure/5ZE3)")
        st.markdown(f"- AlphaFold: https://alphafold.ebi.ac.uk/entry/O60342")
        st.markdown("---")
        st.subheader("Notes")
        st.write("""
        • The app attempts AlphaFold auto-download using common filename patterns (v1/v2, .pdb/.pdb.gz).  
        • NGL.js is embedded for interactive 3D viewing. For very large PDBs you may prefer streaming via URL instead of embedding.  
        • If you want automatic MobiDB/DisProt checks or coloring by pLDDT, I can add that next.
        """)

if __name__ == "__main__":
    main()
