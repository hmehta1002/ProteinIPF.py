import streamlit as st
from io import StringIO
import requests
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np
from Bio import SeqIO

# -----------------------------
# Utility Functions
# -----------------------------
def fetch_alphafold_structure(seq):
    # dummy API call – in practice, map sequence to AlphaFold DB entry
    return {"status":"ok","sequence":seq[:10]}

def scan_motifs_domains(seq):
    motifs = {"N-glycosylation":"N[^P][ST][^P]"}
    matches = {}
    import re
    for name, pattern in motifs.items():
        match_list = [m.start() for m in re.finditer(pattern, seq)]
        if match_list:
            matches[name] = match_list
    return matches

def fetch_ipf_network():
    # example network for IPF (replace with real CSV for production)
    edges = pd.DataFrame({
        "source":["A","A","B","C","D"],
        "target":["B","C","D","E","F"],
        "weight":[1,2,1,1,3]
    })
    G = nx.from_pandas_edgelist(edges,"source","target",["weight"])
    return G

def compute_centrality(G):
    deg = nx.degree_centrality(G)
    clo = nx.closeness_centrality(G)
    bet = nx.betweenness_centrality(G)
    return pd.DataFrame({
        "Node":list(G.nodes()),
        "Degree":[deg[n] for n in G.nodes()],
        "Closeness":[clo[n] for n in G.nodes()],
        "Betweenness":[bet[n] for n in G.nodes()]
    })

def detect_communities(G):
    import community as community_louvain
    partition = community_louvain.best_partition(G)
    return pd.DataFrame(list(partition.items()),columns=["Node","Community"])

# Novel Features
def dmnfz(seq,G):
    hubs=[n for n,d in nx.degree(G) if d>1]
    motifs=scan_motifs_domains(seq)
    return {h:motifs for h in hubs}

def dfwmin(seq,G):
    cent=compute_centrality(G)
    motifs=scan_motifs_domains(seq)
    return {m:cent["Degree"].mean() for m in motifs}

def viz3d(G):
    pos=nx.spring_layout(G,dim=3,seed=42)
    edge_trace=go.Scatter3d(x=[],y=[],z=[],line=dict(width=2,color='blue'),hoverinfo='none',mode='lines')
    for e in G.edges():
        x0,y0,z0=pos[e[0]]
        x1,y1,z1=pos[e[1]]
        edge_trace.x+=tuple([x0,x1,None])
        edge_trace.y+=tuple([y0,y1,None])
        edge_trace.z+=tuple([z0,z1,None])
    node_trace=go.Scatter3d(x=[],y=[],z=[],mode='markers+text',marker=dict(size=8,color='red'),
                            text=list(G.nodes()),textposition="top center")
    for n in G.nodes():
        x,y,z=pos[n]
        node_trace.x+=tuple([x]);node_trace.y+=tuple([y]);node_trace.z+=tuple([z])
    return go.Figure(data=[edge_trace,node_trace])

def seq_struct_map(seq,structure_json):
    motifs=scan_motifs_domains(seq)
    return motifs

def drug_targets(G):
    cent=compute_centrality(G)
    return cent.sort_values("Betweenness",ascending=False).head(5)["Node"].tolist()

def cross_seq(seq,known):
    out={}
    for k,s in known.items():
        matches=sum(1 for a,b in zip(seq,s) if a==b)
        out[k]=matches/len(seq)
    return out

def heatmap_nodes(seq,G):
    cent=compute_centrality(G)
    motifs=scan_motifs_domains(seq)
    data=np.zeros((len(cent),len(motifs)))
    for i,node in enumerate(cent["Node"]):
        for j,m in enumerate(motifs):
            data[i,j]=cent.iloc[i]["Degree"]
    fig=go.Figure(data=go.Heatmap(z=data,
                                  x=list(motifs.keys()),
                                  y=cent["Node"],
                                  colorscale="Viridis"))
    return fig

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Protein-IPF Explorer",layout="wide")
st.title("Protein-IPF Network Explorer")

tabs=st.tabs(["Home","Upload Sequence","AlphaFold","Motif/Domain","IPF Network",
"Centrality","Communities","DMNFZ","DFWMIN","3D Network","Seq-Struct Map",
"Drug Targets","Cross Comparison","Heatmap","Export"])

with tabs[0]:
    st.markdown("Upload a protein sequence and explore IPF-related interactions, motifs, structure, and novel insights.")

seq=""
with tabs[1]:
    f=st.file_uploader("Upload FASTA",type=["fasta","fa"])
    if f:
        fasta_io=StringIO(f.getvalue().decode())
        record=SeqIO.read(fasta_io,"fasta")
        seq=str(record.seq)
        st.code(seq)

with tabs[2]:
    if seq:
        structure_json=fetch_alphafold_structure(seq)
        st.write("AlphaFold:",structure_json)

with tabs[3]:
    if seq:
        motifs=scan_motifs_domains(seq)
        st.dataframe(pd.DataFrame(list(motifs.items()),columns=["Motif","Positions"]))

with tabs[4]:
    G=fetch_ipf_network()
    st.write(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

with tabs[5]:
    cent=compute_centrality(G)
    st.dataframe(cent)

with tabs[6]:
    comm=detect_communities(G)
    st.dataframe(comm)

with tabs[7]:
    if seq:
        st.write(dmnfz(seq,G))

with tabs[8]:
    if seq:
        st.write(dfwmin(seq,G))

with tabs[9]:
    st.plotly_chart(viz3d(G),use_container_width=True)

with tabs[10]:
    if seq:
        st.write(seq_struct_map(seq,{}))

with tabs[11]:
    st.write("Potential Drug Targets:",drug_targets(G))

with tabs[12]:
    known={"IPF1":"MASTSEQ","IPF2":"SEQUENCEEX"}
    if seq:
        st.write(cross_seq(seq,known))

with tabs[13]:
    if seq:
        st.plotly_chart(heatmap_nodes(seq,G),use_container_width=True)

with tabs[14]:
    csv=compute_centrality(G).to_csv(index=False)
    st.download_button("Download Centrality CSV",csv,"centrality.csv")
