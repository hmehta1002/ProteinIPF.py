import re
import json
import math
from typing import List, Dict, Optional, Tuple

import requests
import pandas as pd
import streamlit as st
import py3Dmol

# ==============================
# App Config
# ==============================
st.set_page_config(
    page_title="IPF Protein Structure Lab (v2: Novel, IDR+Ligand, AF2 Fallback)",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧬",
)

st.title("🧬 IPF Protein Structure Lab — v2 (IDR + Ligand-aware, AF2 Fallback)")
st.caption(
    "Novel Streamlit implementation: direct PDB display (no manual downloads), AF pLDDT-driven IDR calling, ligand-contact mapping, variant overlays, and optional dual-model alignment."
)

# ==============================
# Constants / Endpoints
# ==============================
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_ENTRY  = "https://rest.uniprot.org/uniprotkb/{acc}"
RCSB_SEARCH    = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_PDB_FILE  = "https://files.rcsb.org/download/{pdbid}.pdb"
AF_PDB_URL     = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v4.pdb"

# pLDDT color thresholds (AlphaFold semantics)
PLDDT_COLORS = [
    (90, 0x2166AC),   # very high
    (70, 0x67A9CF),   # confident
    (50, 0xD1E5F0),   # low
    (0,  0xFDDBC7),   # very low
]

# ==============================
# Caching: network helpers
# ==============================
@st.cache_data(show_spinner=False, ttl=3600)
def _get(url, params=None, headers=None, timeout=30):
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

@st.cache_data(show_spinner=False, ttl=3600)
def _get_json(url, params=None, headers=None, timeout=30):
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False, ttl=3600)
def _post_json(url, payload, headers=None, timeout=30):
    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ==============================
# UniProt utilities
# ==============================
@st.cache_data(show_spinner=False, ttl=3600)
def get_uniprot_from_query(q: str) -> Optional[Dict]:
    """
    Accepts gene/protein names or UniProt accession. Returns:
    { 'acc', 'gene', 'organism', 'length', 'sequence' }
    """
    if not q:
        return None
    q = q.strip()

    # Accession direct form
    if re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]", q):
        try:
            j = _get_json(UNIPROT_ENTRY.format(acc=q), headers={"Accept":"application/json"})
            return _parse_uniprot_entry(j)
        except Exception:
            pass

    params = {
        "query": q,
        "fields": "accession,organism_name,protein_name,gene_primary,sequence,length",
        "format": "json",
        "size": "5",
    }
    try:
        j = _get_json(UNIPROT_SEARCH, params=params, headers={"Accept":"application/json"})
        if j.get("results"):
            results = j["results"]
            human = [r for r in results if (r.get("organism", {}).get("scientificName","" ).lower() == "homo sapiens")]
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
def get_uniprot_features(acc: str) -> List[Dict]:
    """Fetch UniProt features and normalize for track/site rendering."""
    try:
        j = _get_json(UNIPROT_ENTRY.format(acc=acc), headers={"Accept":"application/json"})
        feats = j.get("features", [])
        keep = {"Domain", "Region", "Repeat", "Transmembrane", "Active site", "Metal binding", "Motif", "Topological domain", "Disulfide bond", "Modified residue", "Binding site"}
        out = []
        for f in feats:
            ftype = f.get("type","")
            loc = f.get("location", {})
            try:
                start = int(loc["start"]["value"])
                end   = int(loc["end"]["value"])
            except Exception:
                continue
            desc = f.get("description") or ftype
            if ftype in keep:
                out.append({"type": ftype, "start": start, "end": end, "label": desc})
        return out
    except Exception:
        return []

# ==============================
# PDB / AF utilities
# ==============================
@st.cache_data(show_spinner=False, ttl=3600)
def search_pdb_for_uniprot(acc: str) -> List[str]:
    """Return a list of PDB IDs mapped to this UniProt accession (most relevant first)."""
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
        return [h["identifier"] for h in hits]
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_pdb_text(pdb_id: str) -> Optional[str]:
    """Download PDB text for analysis (ligand contacts etc.). View still rendered directly via query, not from file."""
    try:
        return _get(RCSB_PDB_FILE.format(pdbid=pdb_id))
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_alphafold_pdb(acc: str) -> Optional[str]:
    try:
        return _get(AF_PDB_URL.format(acc=acc))
    except Exception:
        return None

# ==============================
# Variant parsing
# ==============================
def parse_variants(s: str) -> List[Tuple[int,int]]:
    """Parse strings like '45, 120-125, 201' -> [(45,45),(120,125),(201,201)]"""
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
    out = sorted(set(out))
    return out

# ==============================
# IDR detection
# ==============================
def _mean_b_by_residue(pdb_text: str) -> Dict[int, float]:
    """For AF PDB (B-factors = pLDDT), compute mean B per residue index."""
    res_b = {}
    res_n = {}
    for line in pdb_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        try:
            resi = int(line[22:26])
            b = float(line[60:66])
        except Exception:
            continue
        res_b[resi] = res_b.get(resi, 0.0) + b
        res_n[resi] = res_n.get(resi, 0) + 1
    for k in list(res_b.keys()):
        res_b[k] = res_b[k] / max(res_n[k], 1)
    return res_b

def idr_regions_from_af(pdb_text: str, thr=50, minrun=8) -> List[Tuple[int,int]]:
    """IDRs where mean pLDDT < thr; return merged runs >= minrun."""
    bmap = _mean_b_by_residue(pdb_text)
    if not bmap:
        return []
    residues = sorted(bmap.keys())
    out = []
    run_s = None
    for r in residues:
        if bmap[r] < thr:
            if run_s is None:
                run_s = r
        else:
            if run_s is not None:
                if r-1 - run_s + 1 >= minrun:
                    out.append((run_s, r-1))
                run_s = None
    if run_s is not None and residues[-1] - run_s + 1 >= minrun:
        out.append((run_s, residues[-1]))
    return out

def idr_regions_from_sequence(seq: str, win=15, minrun=10) -> List[Tuple[int,int]]:
    """
    Lightweight fallback: mark windows as disordered if
    - low hydropathy (Kyte-Doolittle crude scale) and
    - enriched in disorder-promoting residues.
    """
    if not seq:
        return []
    kd = {
        'I':4.5,'V':4.2,'L':3.8,'F':2.8,'C':2.5,'M':1.9,'A':1.8,'G':-0.4,'T':-0.7,
        'S':-0.8,'W':-0.9,'Y':-1.3,'P':-1.6,'H':-3.2,'E':-3.5,'Q':-3.5,'D':-3.5,
        'N':-3.5,'K':-3.9,'R':-4.5
    }
    dis_fav = set(list("GSPQEKRDN"))
    n = len(seq)
    flags = [False]*n
    for i in range(n - win + 1):
        frag = seq[i:i+win]
        hyd = sum(kd.get(a,0) for a in frag)/win
        frac_dis = sum(1 for a in frag if a in dis_fav)/win
        if hyd < -0.5 and frac_dis > 0.45:
            for j in range(i, i+win):
                flags[j] = True
    out = []
    i = 0
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            if j - i >= minrun:
                out.append((i+1, j))  # 1-based
            i = j
        else:
            i += 1
    return out

# ==============================
# Ligand pockets / binding features
# ==============================
def _parse_pdb_atoms(pdb_text: str):
    """Minimal PDB parser for protein CA and ligand atoms (excludes water)."""
    protein_atoms = {}
    ligand_atoms = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            try:
                name = line[12:16].strip()
                chain= line[21].strip() or "?"
                resi = int(line[22:26])
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except Exception:
                continue
            key = (chain, resi)
            if name == "CA":
                protein_atoms[key] = (x,y,z,True)
            else:
                if key not in protein_atoms:
                    protein_atoms[key] = (x,y,z,False)
        elif line.startswith("HETATM"):
            try:
                resn = line[17:20].strip()
                if resn in ("HOH", "WAT"):
                    continue
                name = line[12:16].strip()
                chain= line[21].strip() or "?"
                resi = int(line[22:26])
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except Exception:
                continue
            ligand_atoms.append({"atom":name, "resn":resn, "chain":chain, "resi":resi, "x":x, "y":y, "z":z})
    return protein_atoms, ligand_atoms

def _dist2(a,b):
    dx=a[0]-b[0]; dy=a[1]-b[1]; dz=a[2]-b[2]
    return dx*dx + dy*dy + dz*dz

def ligand_contacts_from_pdb(pdb_text: str, cutoff=4.0) -> Dict[str, List[Tuple[str,int]]]:
    prot, ligs = _parse_pdb_atoms(pdb_text)
    if not prot or not ligs:
        return {}
    cutoff2 = cutoff*cutoff
    lig_groups = {}
    for at in ligs:
        key = f"{at['resn']}:{at['chain']}:{at['resi']}"
        lig_groups.setdefault(key, []).append((at["x"], at["y"], at["z"]))
    contacts = {}
    for lkey, latoms in lig_groups.items():
        hit = set()
        for (chain, r), p in prot.items():
            pxyz = (p[0], p[1], p[2])
            for xyz in latoms:
                if _dist2(pxyz, xyz) <= cutoff2:
                    hit.add((chain, r))
                    break
        if hit:
            contacts[lkey] = sorted(list(hit), key=lambda x:(x[0],x[1]))
    return contacts

def binding_residues_from_uniprot_features(features: List[Dict]) -> List[int]:
    resis = set()
    for f in features:
        if f["type"] in ("Binding site", "Metal binding"):
            for r in range(f["start"], f["end"]+1):
                resis.add(r)
    return sorted(resis)

# ==============================
# Visualization helpers (py3Dmol)
# ==============================
def _add_variant_spheres(view, intervals: List[Tuple[int,int]], model_index=None):
    model_sel = {} if model_index is None else {"model": model_index}
    for a,b in intervals:
        sel = dict(model_sel, **{"resi": list(range(a, b+1))})
        view.setStyle(sel, {"stick": {"radius":0.2}})
        mid = a + (b - a)//2
        sel_mid = dict(model_sel, **{"resi":[mid]})
        view.addLabel(f"{a}-{b}" if a!=b else f"{a}",
            {"backgroundOpacity":0.6, "fontSize":10, "alignment":"topLeft"},
            sel_mid
        )
        view.setStyle(sel_mid, {"sphere": {"radius":1.0}})

def _add_idr_overlay(view, idr_intervals: List[Tuple[int,int]], model_index=None):
    model_sel = {} if model_index is None else {"model": model_index}
    for a,b in idr_intervals:
        sel = dict(model_sel, **{"resi": list(range(a, b+1))})
        view.setStyle(sel, {"cartoon": {"color":"purple","opacity":0.6}})

def _add_ligand_contacts_overlay(view, contacts: Dict[str, List[Tuple[str,int]]], model_index=None):
    model_sel = {} if model_index is None else {"model": model_index}
    for lkey, contact_list in contacts.items():
        resn, chain, resi = lkey.split(":")
        try:
            resi = int(resi)
        except:
            continue
        view.setStyle(dict(model_sel, **{"resn":resn, "chain":chain, "resi":[resi]}), {"sphere": {"radius":1.0}})
        for ch, rnum in contact_list:
            view.setStyle(dict(model_sel, **{"chain":ch, "resi":[rnum]}), {"stick":{"radius":0.25}})

# ---- Core renderers ----

def render_pdb_by_id(pdb_id: str,
                     variants: List[Tuple[int,int]],
                     idr_intervals: List[Tuple[int,int]],
                     contacts: Optional[Dict[str, List[Tuple[str,int]]]] = None,
                     binding_resis: Optional[List[int]] = None,
                     height=560):
    """Directly render a PDB entry by ID via py3Dmol's remote fetch (no file download UI)."""
    v = py3Dmol.view(query=f"pdb:{pdb_id}", width=900, height=height)
    v.setStyle({}, {"cartoon":{"color":"spectrum"}})
    if idr_intervals:
        _add_idr_overlay(v, idr_intervals)
    if variants:
        _add_variant_spheres(v, variants)
    if contacts:
        _add_ligand_contacts_overlay(v, contacts)
    elif binding_resis:
        for r in binding_resis:
            v.setStyle({"resi":[int(r)]}, {"stick":{"radius":0.25}})
    v.zoomTo()
    st.components.v1.html(v._make_html(), height=height)


def render_af_from_text(af_pdb_text: str,
                        variants: List[Tuple[int,int]],
                        idr_intervals: List[Tuple[int,int]],
                        binding_resis: Optional[List[int]] = None,
                        height=560):
    """Render AlphaFold PDB (text) with pLDDT coloring, plus overlays."""
    v = py3Dmol.view(width=900, height=height)
    v.addModel(af_pdb_text, "pdb")
    # color cartoon by mean B (pLDDT) per residue
    script = """
    var m=this.getModel();
    var atoms=m.selectedAtoms({});
    var byResi = {};
    atoms.forEach(function(a){ if(!byResi[a.resi]) byResi[a.resi]=[]; byResi[a.resi].push(a); });
    function col(b){ if(b>=90) return 0x2166AC; if(b>=70) return 0x67A9CF; if(b>=50) return 0xD1E5F0; return 0xFDDBC7; }
    for (var res in byResi){
      var arr = byResi[res];
      var mean=0; for (var i=0;i<arr.length;i++){ mean+=arr[i].b; } mean/=arr.length;
      var color=col(mean);
      m.setStyle({resi:parseInt(res)}, {cartoon:{color:color}});
    }
    """
    v.setStyle({}, {"cartoon":{}})
    v.script(script)
    if idr_intervals:
        _add_idr_overlay(v, idr_intervals)
    if variants:
        _add_variant_spheres(v, variants)
    if binding_resis:
        for r in binding_resis:
            v.setStyle({"resi":[int(r)]}, {"stick":{"radius":0.25}})
    v.zoomTo()
    st.components.v1.html(v._make_html(), height=height)


def render_aligned_pdb_and_af(pdb_id: str,
                              af_pdb_text: str,
                              idr_af: List[Tuple[int,int]],
                              binding_resis: Optional[List[int]],
                              variants: List[Tuple[int,int]],
                              height=600):
    """Two models in one viewer, aligned: PDB (spectrum) + AF (pLDDT)."""
    v = py3Dmol.view(width=950, height=height)
    v.addModel(fetch_pdb_text(pdb_id) or "", "pdb")  # ensures same coordinate source for alignment
    v.addModel(af_pdb_text, "pdb")
    v.setStyle({"model":0}, {"cartoon":{"color":"spectrum"}})
    script = """
    var m=this.getModel(1);
    var atoms=m.selectedAtoms({});
    var byResi = {};
    atoms.forEach(function(a){ if(!byResi[a.resi]) byResi[a.resi]=[]; byResi[a.resi].push(a); });
    function col(b){ if(b>=90) return 0x2166AC; if(b>=70) return 0x67A9CF; if(b>=50) return 0xD1E5F0; return 0xFDDBC7; }
    for (var res in byResi){
      var arr = byResi[res];
      var mean=0; for (var i=0;i<arr.length;i++){ mean+=arr[i].b; } mean/=arr.length;
      var color=col(mean);
      m.setStyle({resi:parseInt(res)}, {cartoon:{color:color}});
    }
    this.align();
    """
    v.script(script)
    if idr_af:
        for a,b in idr_af:
            v.setStyle({"model":1, "resi": list(range(a,b+1))}, {"cartoon":{"color":"purple","opacity":0.6}})
    if binding_resis:
        for r in binding_resis:
            v.setStyle({"model":1, "resi":[int(r)]}, {"stick":{"radius":0.25}})
    if variants:
        for a,b in variants:
            v.setStyle({"model":1, "resi": list(range(a,b+1))}, {"stick":{"radius":0.2}})
    v.zoomTo()
    st.components.v1.html(v._make_html(), height=height)

# ==============================
# Feature tracks (Plotly)
# ==============================

def feature_tracks(length: Optional[int], feats: List[Dict]):
    if not length or length <= 0:
        st.info("No sequence length found to render feature tracks.")
        return
    lanes = {}
    for f in feats:
        lanes.setdefault(f["type"], []).append(f)

    st.markdown("**Protein feature tracks (UniProt):**")
    for ftype, arr in lanes.items():
        cols = st.columns([1,6,1])
        with cols[1]:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_shape(type="rect", x0=1, x1=length, y0=0, y1=1, line=dict(color="rgba(0,0,0,0.2)"))
            for f in arr:
                fig.add_shape(type="rect", x0=f["start"], x1=f["end"], y0=0, y1=1)
                fig.add_annotation(x=(f["start"]+f["end"])/2, y=0.5, text=f["label"], showarrow=False, font=dict(size=10))
            fig.update_yaxes(visible=False, range=[0,1])
            fig.update_xaxes(title_text=f"{ftype} (1…{length})", range=[0, length+1], showgrid=False)
            fig.update_layout(height=80, margin=dict(l=10,r=10,t=20,b=10))
            st.plotly_chart(fig, use_container_width=True)

# ==============================
# Sidebar: Input & Controls
# ==============================
with st.sidebar:
    st.header("🔧 Input")
    mode = st.radio("Choose input mode", ["Type one query", "Upload CSV (batch)"])
    st.caption("CSV columns: query[, variants]")

    default_variants = st.text_input("Optional variants (e.g. 87, 120-125)", "")

    st.divider()
    st.header("🎛 Overlays & Options")
    overlay_idr = st.checkbox("Show IDR regions", value=True)
    overlay_ligand = st.checkbox("Show ligand pockets / binding residues", value=True)
    align_when_dual = st.checkbox("Align PDB and AlphaFold (if both exist)", value=True)
    contact_cutoff = st.slider("Ligand contact cutoff (Å)", min_value=3.0, max_value=6.0, value=4.0, step=0.5)

    st.divider()
    # Example CSV
    sample_df = pd.DataFrame({
        "query": ["LOXL2", "MUC5B", "TGFB1", "COL1A1", "SFTPC", "FOXM1"],
        "variants": ["287,320-330", "", "50", "150-170", "82, 101", ""]
    })
    st.download_button("📥 Download sample.csv", data=sample_df.to_csv(index=False).encode(),
                       file_name="sample_proteins.csv", mime="text/csv")

# ==============================
# Core pipeline per query
# ==============================

def process_one(query: str, variants_text: str):
    st.subheader(f"🔎 Query: `{query}`")

    with st.spinner("Resolving UniProt and features..."):
        up = get_uniprot_from_query(query)
    if not up:
        st.error("No UniProt match found.")
        return

    acc   = up["acc"]; gene  = up["gene"]; org = up["organism"]; length = up["length"]; seq = up["sequence"]

    meta_cols = st.columns(5)
    meta_cols[0].metric("UniProt", acc)
    meta_cols[1].metric("Gene", gene or "—")
    meta_cols[2].metric("Organism", org or "—")
    meta_cols[3].metric("Length", length or 0)
    meta_cols[4].metric("Sequence present", "Yes" if seq else "No")

    with st.spinner("Fetching UniProt feature annotations..."):
        feats = get_uniprot_features(acc)
    if feats:
        feature_tracks(length, feats)

    with st.spinner("Searching PDB and AlphaFold models..."):
        pdb_ids = search_pdb_for_uniprot(acc)
        af_text = fetch_alphafold_pdb(acc)

    have_pdb = len(pdb_ids) > 0
    have_af = bool(af_text)

    selected_pdb = None
    pdb_contacts = {}
    if have_pdb:
        # let user choose a PDB if multiple
        selected_pdb = st.selectbox("Select an experimental PDB (if multiple available):", pdb_ids, index=0)
        if overlay_ligand:
            with st.spinner("Analyzing ligand contacts from selected PDB..."):
                pdb_text_for_contacts = fetch_pdb_text(selected_pdb)
                if pdb_text_for_contacts:
                    pdb_contacts = ligand_contacts_from_pdb(pdb_text_for_contacts, cutoff=contact_cutoff)

    intervals = parse_variants(variants_text)

    # Prepare IDR overlays
    idr_af = []
    idr_seq = []
    if overlay_idr:
        if have_af:
            with st.spinner("Calling IDRs from AF pLDDT..."):
                idr_af = idr_regions_from_af(af_text, thr=50, minrun=8)
        else:
            with st.spinner("Heuristic sequence-only IDR call..."):
                idr_seq = idr_regions_from_sequence(seq or "", win=15, minrun=10)

    # Binding residues from UniProt
    up_binding_res = binding_residues_from_uniprot_features(feats) if (feats and overlay_ligand) else []

    # ---- Render cases ----
    if have_pdb and have_af:
        st.success(f"Found **PDB** `{selected_pdb}` and **AlphaFold** model for `{acc}`.")
        tabs = st.tabs(["🧪 Experimental (PDB)", "🧠 AlphaFold (pLDDT)", "🔗 Aligned (PDB + AF)"])

        with tabs[0]:
            idr_for_pdb_overlay = idr_af if idr_af else idr_seq
            render_pdb_by_id(
                selected_pdb,
                variants=intervals,
                idr_intervals=idr_for_pdb_overlay,
                contacts=pdb_contacts,
                binding_resis=up_binding_res if not pdb_contacts else None,
                height=560,
            )

        with tabs[1]:
            st.caption("**pLDDT legend**: ≥90 (very high), 70–89 (confident), 50–69 (low), <50 (very low)")
            render_af_from_text(
                af_pdb_text=af_text,
                variants=intervals,
                idr_intervals=idr_af,
                binding_resis=up_binding_res,
                height=560,
            )

        with tabs[2]:
            if align_when_dual:
                render_aligned_pdb_and_af(
                    pdb_id=selected_pdb,
                    af_pdb_text=af_text,
                    idr_af=idr_af,
                    binding_resis=up_binding_res,
                    variants=intervals,
                    height=600,
                )
            else:
                st.info("Alignment toggled off in sidebar.")

    elif have_pdb:
        st.success(f"Found **PDB** `{selected_pdb}` for `{acc}`.")
        idr_for_pdb_overlay = idr_seq  # if no AF, fall back to sequence-based IDR
        render_pdb_by_id(
            selected_pdb,
            variants=intervals,
            idr_intervals=idr_for_pdb_overlay,
            contacts=pdb_contacts,
            binding_resis=up_binding_res if not pdb_contacts else None,
            height=560,
        )

    elif have_af:
        st.warning(f"No experimental PDB found for `{acc}` — showing **AlphaFold** model.")
        st.caption("**pLDDT legend**: ≥90 (very high), 70–89 (confident), 50–69 (low), <50 (very low)")
        render_af_from_text(
            af_pdb_text=af_text,
            variants=intervals,
            idr_intervals=idr_af,
            binding_resis=up_binding_res,
            height=560,
        )
    else:
        st.error("Neither PDB nor AlphaFold model could be retrieved. Try another ID, or check network access.")

# ==============================
# Run pipeline (single or batch)
# ==============================
if mode == "Type one query":
    q = st.text_input("Enter a gene/protein name or UniProt ID (e.g., LOXL2 or Q9Y4K0)", "LOXL2")
    variants_in = st.text_input("Optional variants for highlighting (e.g., 287, 320-330)", st.session_state.get("variants_one", ""))
    if st.button("Analyze", type="primary"):
        st.session_state["variants_one"] = variants_in
        process_one(q, variants_in)
else:
    up = st.file_uploader("Upload CSV with columns: query[, variants]", type=["csv"])
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            df = None
        if df is not None:
            if "query" not in df.columns:
                st.error("CSV must contain a column named 'query'. Optional column: 'variants'.")
            else:
                st.success(f"Loaded {len(df)} queries.")
                for i, row in df.iterrows():
                    q = str(row["query"]).strip()
                    var = str(row["variants"]).strip() if "variants" in df.columns and not pd.isna(row["variants"]) else ""
                    with st.container(border=True):
                        process_one(q, var)
                    st.divider()

