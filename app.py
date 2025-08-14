# app.py
import io
import re
import json
import math
import time
import requests
import pandas as pd
import streamlit as st
import py3Dmol
from typing import List, Dict, Optional, Tuple

st.set_page_config(
    page_title="IPF Protein Structure Lab (Ultra-Novel)",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬",
)

st.title("🧬 IPF Protein Structure Lab — Novel ID-Aware, Variant-Aware, AF2-Fallback")

st.markdown("""
**What’s special here?**  
- Type a **gene/protein name** (e.g., `LOXL2`, `MUC5B`, `TGFB1`) or a **UniProt ID** (`Q9Y4K0`), or upload a CSV with column **`query`** (and optional **`variants`** like `87,120-125`).
- We’ll:  
  1) map your query → **UniProt**,  
  2) search **PDB** (experimental),  
  3) fall back to **AlphaFold** if no PDB,  
  4) color AF by **pLDDT**,  
  5) render **feature tracks** (domains/active-sites),  
  6) optionally **highlight residues/intervals** you provide.
""")

# -----------------------------
# Config & Constants
# -----------------------------
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY  = "https://rest.uniprot.org/uniprotkb/{acc}"
RCSB_SEARCH    = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_PDB_FILE  = "https://files.rcsb.org/download/{pdbid}.pdb"
AF_PDB_URL     = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb"

# pLDDT color thresholds (same semantics as AF DB)
PLDDT_COLORS = [
    (90, "0x2166AC"),   # very high
    (70, "0x67A9CF"),   # confident
    (50, "0xD1E5F0"),   # low
    (0,  "0xFDDBC7"),   # very low
]

# -----------------------------
# Caching helpers
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def _get(url, params=None, headers=None, timeout=30):
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

@st.cache_data(show_spinner=False, ttl=3600)
def _post_json(url, payload, headers=None, timeout=30):
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=3600)
def get_uniprot_from_query(q: str) -> Optional[Dict]:
    """
    Accepts gene/protein names or UniProt accession. Returns a dict:
    { 'acc': 'Qxxxx', 'gene': 'SYMBOL', 'organism': 'Homo sapiens', 'length': int, 'sequence': '...' }
    """
    q = q.strip()
    # If looks like accession, try direct fetch first
    if re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]", q):
        try:
            txt = _get(UNIPROT_ENTRY.format(acc=q), headers={"Accept":"application/json"})
            j = json.loads(txt)
            return _parse_uniprot_entry(j)
        except Exception:
            pass

    # Otherwise search
    params = {
        "query": q,
        "fields": "accession,organism_name,protein_name,gene_primary,sequence,length",
        "format": "json",
        "size": "5",
    }
    try:
        txt = _get(UNIPROT_SEARCH, params=params, headers={"Accept":"application/json"})
        j = json.loads(txt)
        if j.get("results"):
            # Prefer human entries if present
            results = j["results"]
            human = [r for r in results if (r.get("organism", {}).get("scientificName","").lower() == "homo sapiens")]
            best = human[0] if human else results[0]
            return _parse_uniprot_entry(best)
    except Exception:
        return None
    return None

def _parse_uniprot_entry(entry_json) -> Dict:
    acc = entry_json["primaryAccession"]
    organism = entry_json.get("organism", {}).get("scientificName", "")
    genes = entry_json.get("genes", [])
    gene_primary = ""
    if genes and isinstance(genes, list):
        gn = genes[0].get("geneName", {}).get("value")
        gene_primary = gn or ""
    seq = entry_json.get("sequence", {}).get("value", "")
    length = entry_json.get("sequence", {}).get("length", None)
    return {"acc": acc, "gene": gene_primary, "organism": organism, "length": length, "sequence": seq}

@st.cache_data(show_spinner=False, ttl=3600)
def search_pdb_for_uniprot(acc: str) -> List[Dict]:
    """
    Search PDB entries linked to this UniProt accession.
    Returns list of dicts with 'pdb_id' and (if available) 'resolution'.
    """
    q = {
      "query": {
        "type": "terminal",
        "service": "text",
        "parameters": {
          "attribute": "rcsb_polymer_entity_container_identifiers.uniprot_accession",
          "operator": "exact_match",
          "value": acc
        }
      },
      "return_type": "entry",
      "request_options": {"return_all_hits": True}
    }
    try:
        j = _post_json(RCSB_SEARCH, q, headers={"Content-Type":"application/json"})
        hits = j.get("result_set", [])
        out = []
        for h in hits:
            pdbid = h["identifier"]
            out.append({"pdb_id": pdbid})
        return out
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=3600)
def download_pdb(pdb_id: str) -> Optional[str]:
    try:
        return _get(RCSB_PDB_FILE.format(pdbid=pdb_id))
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def download_alphafold_pdb(acc: str) -> Optional[str]:
    try:
        return _get(AF_PDB_URL.format(acc=acc))
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def get_uniprot_features(acc: str) -> List[Dict]:
    """
    Fetch UniProt features for track rendering.
    """
    try:
        txt = _get(UNIPROT_ENTRY.format(acc=acc), headers={"Accept":"application/json"})
        j = json.loads(txt)
        feats = j.get("features", [])
        # filter for some key types to keep it clean
        keep = {"Domain", "Region", "Repeat", "Transmembrane", "Active site", "Metal binding", "Motif", "Topological domain", "Disulfide bond", "Modified residue", "Binding site"}
        out = []
        for f in feats:
            ftype = f.get("type","")
            if ftype in keep:
                loc = f.get("location", {})
                try:
                    start = int(loc["start"]["value"])
                    end   = int(loc["end"]["value"])
                except Exception:
                    continue
                label = f.get("description") or ftype
                out.append({"type": ftype, "start": start, "end": end, "label": label})
        return out
    except Exception:
        return []

# -----------------------------
# Variant parsing
# -----------------------------
def parse_variants(s: str) -> List[Tuple[int,int]]:
    """
    Parse strings like '45, 120-125, 201' -> [(45,45),(120,125),(201,201)]
    """
    if not s or not isinstance(s, str):
        return []
    parts = re.split(r"[,\s;]+", s.strip())
    out = []
    for p in parts:
        if not p:
            continue
        if "-" in p:
            a,b = p.split("-",1)
            if a.isdigit() and b.isdigit():
                out.append((int(a), int(b)))
        else:
            if p.isdigit():
                pos = int(p)
                out.append((pos, pos))
    # dedupe & sort
    out = sorted(set(out))
    return out

# -----------------------------
# Visualization helpers
# -----------------------------
def _plddt_color(bval: float) -> str:
    for thr, col in PLDDT_COLORS:
        if bval >= thr:
            return col
    return "0xDDDDDD"

def _add_variant_spheres(view, intervals: List[Tuple[int,int]]):
    for a,b in intervals:
        sel = {"resi": list(range(a, b+1))}
        view.addStyle(sel, {"stick": {"radius":0.2}})
        # put sphere at middle residue
        mid = a + (b - a)//2
        view.addStyle({"resi": [mid]}, {"sphere": {"radius":1.0}})
        view.addLabel(f"{a}-{b}" if a!=b else f"{a}",
            {"backgroundOpacity":0.6, "fontSize":10, "alignment":"topLeft"},
            {"resi":[mid]}
        )

def show_structure(pdb_str: str, title: str, is_alphafold: bool, variants: List[Tuple[int,int]], height=520):
    """
    Render a structure; if AlphaFold, color by pLDDT using B-factors (stored in AF PDB).
    """
    view = py3Dmol.view(width=800, height=height)
    view.addModel(pdb_str, "pdb")

    if is_alphafold:
        # Color by B-factor (pLDDT). We can do an approximate per-residue mapping by atom.
        # py3Dmol JS API supports conditional coloring by b factor via setStyle + colorfunc
        # In python wrapper, we’ll loop segments with atom-level selection in JS.
        script = """
        var m=this.getModel();
        var atoms=m.selectedAtoms({});
        var byResi = {};
        atoms.forEach(function(a){
          if(!byResi[a.resi]) byResi[a.resi]=[];
          byResi[a.resi].push(a);
        });
        function col(b){
          if(b>=90) return 0x2166AC;
          if(b>=70) return 0x67A9CF;
          if(b>=50) return 0xD1E5F0;
          return 0xFDDBC7;
        }
        for (var res in byResi){
          var arr = byResi[res];
          var mean=0;
          for (var i=0;i<arr.length;i++){ mean+=arr[i].b; }
          mean/=arr.length;
          var color=col(mean);
          m.setStyle({resi:parseInt(res)}, {cartoon:{color:color}});
        }
        """
        view.setStyle({}, {"cartoon":{}})  # default first, then recolor
        view.zoomTo()
        view.script(script)
    else:
        view.setStyle({}, {"cartoon":{"color":"spectrum"}})
        view.zoomTo()

    if variants:
        _add_variant_spheres(view, variants)

    view.zoomTo()
    return view

def feature_tracks(length: Optional[int], feats: List[Dict]):
    if not length or length <= 0:
        st.info("No sequence length found to render feature tracks.")
        return
    # Build lanes by feature type
    lanes = {}
    for f in feats:
        lanes.setdefault(f["type"], []).append(f)

    st.markdown("**Protein feature tracks (UniProt):**")
    for ftype, arr in lanes.items():
        # render a simple horizontal bar representing sequence, then blocks
        cols = st.columns([1,6,1])
        with cols[1]:
            import plotly.graph_objects as go
            fig = go.Figure()
            # baseline
            fig.add_shape(type="rect", x0=1, x1=length, y0=0, y1=1, line=dict(color="rgba(0,0,0,0.2)"))
            # features
            for f in arr:
                fig.add_shape(type="rect", x0=f["start"], x1=f["end"], y0=0, y1=1)
                fig.add_annotation(x=(f["start"]+f["end"])/2, y=0.5, text=f["label"], showarrow=False, font=dict(size=10))
            fig.update_yaxes(visible=False, range=[0,1])
            fig.update_xaxes(title_text=f"{ftype} (1…{length})", range=[0, length+1], showgrid=False)
            fig.update_layout(height=80, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Sidebar: Input
# -----------------------------
with st.sidebar:
    st.header("🔧 Input")
    mode = st.radio("Choose input mode", ["Type one query", "Upload CSV (batch)"])
    default_variants = st.text_input("Optional variants (e.g. 87, 120-125)", "")
    align_when_dual = st.checkbox("Align PDB and AlphaFold (if both exist)", value=True)
    st.divider()
    # Sample CSV
    sample_df = pd.DataFrame({
        "query": ["LOXL2", "MUC5B", "TGFB1", "COL1A1", "SFTPC", "FOXM1"],
        "variants": ["287,320-330", "", "50", "150-170", "82, 101", ""]
    })
    sample_csv = sample_df.to_csv(index=False).encode()
    st.download_button("📥 Download sample.csv", data=sample_csv, file_name="sample_proteins.csv", mime="text/csv")

# -----------------------------
# Main logic
# -----------------------------
def process_one(query: str, variants_text: str):
    st.subheader(f"🔎 Query: `{query}`")
    up = get_uniprot_from_query(query)
    if not up:
        st.error("No UniProt match found.")
        return

    acc   = up["acc"]
    gene  = up["gene"]
    org   = up["organism"]
    length = up["length"]

    meta_cols = st.columns(4)
    meta_cols[0].metric("UniProt", acc)
    meta_cols[1].metric("Gene", gene or "—")
    meta_cols[2].metric("Organism", org or "—")
    meta_cols[3].metric("Length", length or 0)

    feats = get_uniprot_features(acc)
    if feats:
        feature_tracks(length, feats)

    # PDB search
    pdb_hits = search_pdb_for_uniprot(acc)
    pdb_id = pdb_hits[0]["pdb_id"] if pdb_hits else None
    have_pdb = False
    pdb_text = None
    if pdb_id:
        pdb_text = download_pdb(pdb_id)
        have_pdb = bool(pdb_text)

    # AlphaFold fallback
    af_text = download_alphafold_pdb(acc)
    have_af = bool(af_text)

    intervals = parse_variants(variants_text)

    if have_pdb and have_af:
        st.success(f"Found **PDB** entry `{pdb_id}` and **AlphaFold** model for `{acc}`.")
        tabs = st.tabs(["🧪 Experimental (PDB)", "🧠 AlphaFold (pLDDT)"])
        with tabs[0]:
            view = show_structure(pdb_text, f"PDB: {pdb_id}", is_alphafold=False, variants=intervals)
            st.components.v1.html(view._make_html(), height=540)
            st.download_button("⬇️ Download PDB file", data=pdb_text, file_name=f"{pdb_id}.pdb")
        with tabs[1]:
            st.caption("**pLDDT legend**: ≥90 (very high), 70–89 (confident), 50–69 (low), <50 (very low)")
            view2 = show_structure(af_text, f"AF: {acc}", is_alphafold=True, variants=intervals)
            # Optional align: load both & align in a composite viewer
            if align_when_dual:
                combo = py3Dmol.view(width=800, height=540)
                combo.addModel(pdb_text, "pdb")
                combo.addModel(af_text, "pdb")
                # set different styles to distinguish
                combo.setStyle({"model":0}, {"cartoon":{"color":"spectrum"}})
                # AF colored by pLDDT (reuse javascript script)
                script = """
                var m=this.getModel(1);
                var atoms=m.selectedAtoms({});
                var byResi = {};
                atoms.forEach(function(a){
                  if(!byResi[a.resi]) byResi[a.resi]=[];
                  byResi[a.resi].push(a);
                });
                function col(b){
                  if(b>=90) return 0x2166AC;
                  if(b>=70) return 0x67A9CF;
                  if(b>=50) return 0xD1E5F0;
                  return 0xFDDBC7;
                }
                for (var res in byResi){
                  var arr = byResi[res];
                  var mean=0;
                  for (var i=0;i<arr.length;i++){ mean+=arr[i].b; }
                  mean/=arr.length;
                  var color=col(mean);
                  m.setStyle({resi:parseInt(res)}, {cartoon:{color:color}});
                }
                this.align()
                """
                combo.script(script)
                if intervals:
                    # add variant spheres on AF model (index 1)
                    for a,b in intervals:
                        combo.addStyle({"model":1, "resi":list(range(a,b+1))}, {"stick":{"radius":0.2}})
                        mid=a+(b-a)//2
                        combo.addStyle({"model":1, "resi":[mid]}, {"sphere":{"radius":1.0}})
                combo.zoomTo()
                st.components.v1.html(combo._make_html(), height=560)
                st.caption("Aligned view: Model 0 = PDB (spectrum), Model 1 = AlphaFold (pLDDT).")
            else:
                st.components.v1.html(view2._make_html(), height=540)
            st.download_button("⬇️ Download AlphaFold PDB", data=af_text, file_name=f"AF-{acc}.pdb")

    elif have_pdb:
        st.success(f"Found **PDB** entry `{pdb_id}` for `{acc}`.")
        view = show_structure(pdb_text, f"PDB: {pdb_id}", is_alphafold=False, variants=intervals)
        st.components.v1.html(view._make_html(), height=540)
        st.download_button("⬇️ Download PDB file", data=pdb_text, file_name=f"{pdb_id}.pdb")

    elif have_af:
        st.warning(f"No experimental PDB found for `{acc}` — showing **AlphaFold** model.")
        st.caption("**pLDDT legend**: ≥90 (very high), 70–89 (confident), 50–69 (low), <50 (very low)")
        view = show_structure(af_text, f"AF: {acc}", is_alphafold=True, variants=intervals)
        st.components.v1.html(view._make_html(), height=540)
        st.download_button("⬇️ Download AlphaFold PDB", data=af_text, file_name=f"AF-{acc}.pdb")
    else:
        st.error("Neither PDB nor AlphaFold model could be retrieved. Try another ID, or check network access.")

# -----------------------------
# Input controls
# -----------------------------
if mode == "Type one query":
    q = st.text_input("Enter a gene/protein name or UniProt ID (e.g., LOXL2 or Q9Y4K0)", "LOXL2")
    variants_in = st.text_input("Optional variants for highlighting (e.g., 287, 320-330)", st.session_state.get("variants_one", ""))
    if st.button("Analyze"):
        st.session_state["variants_one"] = variants_in
        process_one(q, variants_in)

else:
    up = st.file_uploader("Upload CSV with columns: query[, variants]", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        if "query" not in df.columns:
            st.error("CSV must contain a column named 'query'. Optional column: 'variants'.")
        else:
            st.success(f"Loaded {len(df)} queries.")
            for i, row in df.iterrows():
                q = str(row["query"]).strip()
                var = str(row["variants"]).strip() if "variants" in df.columns and not pd.isna(row["variants"]) else ""
                process_one(q, var)
                st.divider()

                st.error(f"No structure found for {protein_choice} in PDB or AlphaFold.")
