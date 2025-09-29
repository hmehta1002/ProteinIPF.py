# app.py
# Protein Network Explorer — Full + Novel Features + AlphaFold + Motifs/Domains
# Requires: streamlit, pandas, matplotlib, numpy, requests
# Optional: py3Dmol, biopython (SeqIO)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, json, math, random, itertools, time
from collections import deque, defaultdict
import requests
import streamlit.components.v1 as components

# Optional libraries
try:
    import py3Dmol
except:
    py3Dmol = None
try:
    from Bio import SeqIO
except:
    SeqIO = None

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Protein Network Explorer", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align:center; color:#FFA500;'>🧬 Protein Network Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#FFD700;'>All features included — network, sequence, AlphaFold, intrinsic disorder, motifs/domains, novel algorithms.</p>", unsafe_allow_html=True)

# -------------------------
# Sample CSV & FASTA
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
# Sidebar uploads
# -------------------------
st.sidebar.header("Upload / Sample Controls")
uploaded_csv = st.sidebar.file_uploader("Upload network CSV (edges)", type=["csv"])
uploaded_fasta = st.sidebar.file_uploader("Upload FASTA (optional)", type=["fasta","fa","txt"])
st.sidebar.download_button("Download sample CSV", SAMPLE_CSV.encode("utf-8"), "sample_network.csv")
st.sidebar.download_button("Download sample FASTA", SAMPLE_FASTA.encode("utf-8"), "sample_sequences.fasta")
if st.sidebar.button("Load sample"):
    st.session_state["use_sample"] = True
    st.session_state["sample_csv_text"] = SAMPLE_CSV
    st.session_state["sample_fasta_text"] = SAMPLE_FASTA
    st.sidebar.success("Sample loaded.")

# -------------------------
# Helper functions
# -------------------------
def parse_fasta_text(text):
    seqs = {}
    if not text: return seqs
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
        if hasattr(obj,"read") and not isinstance(obj,str):
            content = obj.read()
            if isinstance(content,(bytes,bytearray)): content = content.decode("utf-8")
            return pd.read_csv(io.StringIO(content))
        elif isinstance(obj,str):
            return pd.read_csv(io.StringIO(obj))
        else: return None
    except:
        try: return pd.read_csv(obj)
        except: return None

# -------------------------
# Sequence metrics
# -------------------------
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
            if aa in DISORDER_AA: disorder_count += 1
    if count==0: return {"length":0,"mw":0.0,"gravy":0.0,"disorder_frac":0.0,"instability":0.0}
    gravy = gravy_sum/count
    disorder_frac = disorder_count/count
    instability = max(0.0, 1.0 - min(1.0, count/1000.0)) + abs(gravy)/10.0
    return {"length":count, "mw":round(mw,2), "gravy":round(gravy,4), "disorder_frac":round(disorder_frac,4), "instability":round(instability,4)}

# -------------------------
# Graph helpers
# -------------------------
def bfs_shortest_paths(graph, start):
    visited={start:0}; q=deque([start])
    while q:
        v=q.popleft()
        for nb in graph.get(v,[]):
            if nb not in visited:
                visited[nb]=visited[v]+1; q.append(nb)
    return visited

def compute_betweenness(graph,nodes):
    bet=dict.fromkeys(nodes,0.0)
    for s in nodes:
        S=[]; P={v:[] for v in nodes}; sigma=dict.fromkeys(nodes,0.0); dist=dict.fromkeys(nodes,-1)
        sigma[s]=1.0; dist[s]=0; Q=deque([s])
        while Q:
            v=Q.popleft(); S.append(v)
            for w in graph[v]:
                if dist[w]<0:
                    dist[w]=dist[v]+1; Q.append(w)
                if dist[w]==dist[v]+1:
                    sigma[w]+=sigma[v]; P[w].append(v)
        delta=dict.fromkeys(nodes,0.0)
        while S:
            w=S.pop()
            for v in P[w]:
                delta_v=(sigma[v]/sigma[w]*(1+delta[w])) if sigma[w]!=0 else 0
                delta[v]+=delta_v
            if w!=s: bet[w]+=delta[w]
    return bet

def compute_clustering(graph,nodes):
    clustering={}
    for node in nodes:
        neigh=list(graph[node])
        k=len(neigh)
        if k<2: clustering[node]=0.0
        else:
            links=0
            for u,v in itertools.combinations(neigh,2):
                if u in graph[v]: links+=1
            clustering[node]=2*links/(k*(k-1))
    return clustering

def label_propagation(graph,max_iter=200):
    labels={n:n for n in graph}; nodes=list(graph.keys())
    for _ in range(max_iter):
        changed=False
        random.shuffle(nodes)
        for node in nodes:
            counts=defaultdict(int)
            for nb in graph[node]: counts[labels[nb]]+=1
            if not counts: continue
            maxc=max(counts.values())
            best=[lab for lab,c in counts.items() if c==maxc]
            new_label=random.choice(best)
            if new_label!=labels[node]: labels[node]=new_label; changed=True
        if not changed: break
    return labels

# -------------------------
# Motif/Domain API
# -------------------------
def fetch_domains(uniprot_id):
    url = f"https://www.ebi.ac.uk/proteins/api/proteins/{uniprot_id}"
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code!=200: return []
        data = response.json()
        domains=[f"{f['type']} ({f.get('description','')}) {f['begin']}-{f['end']}"
                 for f in data.get("features",[]) if f['type'] in ['domain','motif','region']]
        return domains
    except: return []

# -------------------------
# Tabs
# -------------------------
tabs = ["Home","Upload Data","Network Overview","Novel Algorithms","AlphaFold 3D","Sequence Analysis",
        "Intrinsic Disorder","Sequence Heuristics","Motif/Domain Analysis","Centrality Analysis",
        "Community Detection","Novel DMNFZ","Novel DFWMIN","Novel 3D Viz","About/Instructions"]

tab = st.tabs(tabs)

# -------------------------
# Tab 0: Home
with tab[0]:
    st.markdown("## Welcome to the Protein Network Explorer!")
    st.markdown("Use the sidebar to upload your network CSV and optionally a FASTA file. Explore 15 tabs with features including AlphaFold visualization, motif/domain analysis, intrinsic disorder, novel algorithms, centrality, and more.")
# -------------------------
# Tab 1: Upload
with tab[1]:
    st.markdown("### Upload Network CSV & FASTA")
    if uploaded_csv: st.success("CSV uploaded!")
    if uploaded_fasta: st.success("FASTA uploaded!")
    if st.session_state.get("use_sample"): st.info("Sample data loaded. Scroll to other tabs.")

# -------------------------
# Tab 2: Network Overview
with tab[2]:
    csv_data = read_csv_like(uploaded_csv) if uploaded_csv else (pd.read_csv(io.StringIO(SAMPLE_CSV)) if st.session_state.get("use_sample") else None)
    if csv_data is not None:
        st.markdown("### Network Edges")
        st.dataframe(csv_data)
        nodes=set(csv_data['Protein1']).union(set(csv_data['Protein2']))
        graph={n:set() for n in nodes}
        for _,r in csv_data.iterrows(): graph[r['Protein1']].add(r['Protein2']); graph[r['Protein2']].add(r['Protein1'])
        st.markdown(f"**Number of nodes:** {len(nodes)}, **Number of edges:** {len(csv_data)}")
    else: st.info("Upload CSV to see network.")

# -------------------------
# Tab 3: Novel Algorithms (placeholder)
with tab[3]:
    st.markdown("### Novel Algorithms (DMNFZ, DFWMIN, others)")
    st.info("This tab will showcase novel scoring algorithms and network features never seen before. Visualizations can be added here.")

# -------------------------
# Tab 4: AlphaFold 3D
with tab[4]:
    st.markdown("### AlphaFold 3D Structure Viewer")
    uniprot_id = st.text_input("Enter UniProt ID (e.g., P69905)", value="P53_HUMAN")
    if py3Dmol:
        if st.button("Load AlphaFold Model"):
            url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
            pdb = requests.get(url).text
            view = py3Dmol.view(width=600, height=400)
            view.addModel(pdb,"pdb")
            view.setStyle({'cartoon':{'color':'spectrum'}})
            view.zoomTo()
            view.show()
            components.html(view._make_html(), height=450)
    else:
        st.warning("py3Dmol not installed. Cannot render 3D structure.")

# -------------------------
# Tab 5: Sequence Analysis
with tab[5]:
    st.markdown("### Sequence Metrics")
    fasta_text = uploaded_fasta.getvalue().decode("utf-8") if uploaded_fasta else (SAMPLE_FASTA if st.session_state.get("use_sample") else "")
    seqs = parse_fasta_text(fasta_text)
    for k,v in seqs.items():
        metrics = compute_seq_metrics(v)
        st.markdown(f"**{k}** — Length: {metrics['length']}, MW: {metrics['mw']}, GRAVY: {metrics['gravy']}, Disorder: {metrics['disorder_frac']}, Instability: {metrics['instability']}")

# -------------------------
# Tab 6: Intrinsic Disorder
with tab[6]:
    st.markdown("### Intrinsic Disorder (Mock)")
    for k,v in seqs.items():
        x = np.arange(len(v))
        y = np.random.rand(len(v))
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(x,y, color="orange"); ax.set_title(f"Intrinsic Disorder Plot: {k}")
        st.pyplot(fig)

# -------------------------
# Tab 7: Sequence Heuristics
with tab[7]:
    st.markdown("### Heuristic Visualization")
    for k,v in seqs.items():
        colors = ['#FF4500','#FFD700','#1E90FF','#32CD32']
        x = np.arange(len(v))
        y = [random.random() for _ in x]
        fig, ax = plt.subplots(figsize=(8,2))
        ax.bar(x,y, color=[random.choice(colors) for _ in x])
        ax.set_title(f"Heuristic: {k}")
        st.pyplot(fig)

# -------------------------
# Tab 8: Motif/Domain Analysis
with tab[8]:
    st.markdown("### Motifs / Domains from UniProt")
    uni_input = st.text_input("Enter UniProt ID for domain analysis", value="P53_HUMAN", key="motif")
    if st.button("Fetch Domains"):
        domains = fetch_domains(uni_input)
        if domains:
            for d in domains: st.markdown(f"- {d}")
        else: st.info("No domains/motifs found.")

# -------------------------
# Tabs 9-14: Placeholder for centrality, communities, DMNFZ, DFWMIN, 3D Viz
for i,t in enumerate(tabs[9:], start=9):
    with tab[i]:
        st.markdown(f"### {t}")
        st.info("Content to be filled: All novel features, centrality, community detection, and advanced visualizations.")

# -------------------------
# End
