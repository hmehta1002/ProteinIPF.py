import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import py3Dmol
import requests
import time
from networkx.algorithms import community

st.set_page_config(page_title="IPF Disorder-Weighted PPI Explorer", layout="wide")

# ---------------------------- Helper Functions ----------------------------
@st.cache_data
def fetch_uniprot_info(protein_id):
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    resp = requests.get(url)
    time.sleep(0.2)
    if resp.status_code==200:
        return resp.json()
    return None

@st.cache_data
def fetch_alphafold_structure(protein_id):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v4.pdb"
    resp = requests.get(url)
    time.sleep(0.5)
    if resp.status_code==200:
        return resp.text
    return None

@st.cache_data
def fetch_string_edges(proteins):
    # Fully connected edges for demo
    edges = []
    for i,p1 in enumerate(proteins):
        for p2 in proteins[i+1:]:
            edges.append({'Protein1': p1, 'Protein2': p2})
    return pd.DataFrame(edges)

def compute_disorder(seq):
    aa_hydro = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
                'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,
                'Y':-1.3,'V':4.2}
    return np.array([max(min((aa_hydro.get(aa,0)+4.5)/9,1),0) for aa in seq])

def compute_vulnerable_regions(disorder_scores, snp_positions):
    return [i for i, s in enumerate(disorder_scores) if s>0.6 and i in snp_positions]

def compute_STKDD(disorder_avg, centrality, num_vulnerable):
    return disorder_avg * centrality * (1 + num_vulnerable)

def compute_DMNFZ(G, disorder_dict):
    dmnfz = {}
    for n in G.nodes():
        neighs = list(G.neighbors(n))
        if len(neighs)==0:
            dmnfz[n]=0
            continue
        neighbor_disorders = [disorder_dict.get(neigh,0) for neigh in neighs]
        mean = np.mean(neighbor_disorders)
        std = np.std(neighbor_disorders) if np.std(neighbor_disorders)>0 else 1
        dmnfz[n] = (disorder_dict.get(n,0)-mean)/std
    return dmnfz

def compute_DFWMIN(G, disorder_dict, stkdd_dict):
    flux_dict = {}
    for u,v in G.edges():
        flux_dict[(u,v)] = ((disorder_dict.get(u,0)+disorder_dict.get(v,0))/2) * ((stkdd_dict.get(u,0)+stkdd_dict.get(v,0))/2)
    return flux_dict

def plot_network(G, pos, metric_dict, metric_name="STKDD", highlight_nodes=[]):
    edge_x, edge_y = [], []
    for u,v in G.edges():
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1,color='#888'),
                            hoverinfo='none', mode='lines')
    node_x, node_y, node_size, node_color, node_text = [],[],[],[],[]
    for n in G.nodes():
        x,y = pos[n]; node_x.append(x); node_y.append(y)
        node_size.append(20 + metric_dict.get(n,0)*30)
        node_color.append(metric_dict.get(n,0))
        node_text.append(f"{n}<br>{metric_name}: {metric_dict.get(n,0):.2f}")
    marker_dict = dict(color=node_color, colorscale='Viridis', size=node_size, colorbar=dict(title=metric_name))
    if highlight_nodes:
        marker_dict['line'] = dict(width=[3 if n in highlight_nodes else 0 for n in G.nodes()], color='red')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
                            textposition="top center", hoverinfo='text', marker=marker_dict)
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False)
    return fig

# ---------------------------- Tabs ----------------------------
tabs = st.tabs([
    "Upload / Files", "STRING Network", "Network Map", "Network Metrics",
    "Protein Details", "Sequences / FASTA", "Motifs / Domains", "3D Viewer",
    "Disorder & Stability", "DMNFZ Explorer", "DFWMIN Explorer",
    "Centrality Measures", "Community Detection", "Network Robustness",
    "Seq-Structure Mapping", "Downloads / Exports"
])

# ---------------------------- Tab 1: Upload ----------------------------
with tabs[0]:
    st.header("Upload Protein FASTA")
    uploaded_fasta = st.file_uploader("Upload FASTA sequences", type=["fasta"])
    if uploaded_fasta:
        seqs = uploaded_fasta.read().decode().split('>')
        seq_dict = {}
        for s in seqs[1:]:
            lines = s.split('\n')
            header = lines[0].strip()
            sequence = ''.join(lines[1:]).strip()
            seq_dict[header] = sequence
        st.session_state['seq_dict'] = seq_dict
        st.success("FASTA sequences loaded")
        st.dataframe(pd.DataFrame({'Protein': list(seq_dict.keys()), 'Length':[len(seq) for seq in seq_dict.values()]}))

# ---------------------------- Tab 2: STRING Network ----------------------------
with tabs[1]:
    st.header("Build PPI Network from STRING")
    if 'seq_dict' in st.session_state:
        proteins = list(st.session_state['seq_dict'].keys())
        df_edges = fetch_string_edges(proteins)
        st.session_state['df_edges'] = df_edges
        G = nx.from_pandas_edgelist(df_edges, 'Protein1', 'Protein2')
        st.session_state['G'] = G
        st.success("PPI network built")
        st.dataframe(df_edges)

# ---------------------------- Tab 3: Network Map ----------------------------
with tabs[2]:
    st.header("Network Map")
    if 'G' in st.session_state:
        G = st.session_state['G']
        seq_dict = st.session_state['seq_dict']
        disorder_dict = {n: np.mean(compute_disorder(seq_dict.get(n,''))) if seq_dict.get(n,'') else np.random.rand() for n in G.nodes()}
        st.session_state['disorder_dict'] = disorder_dict
        eig = nx.eigenvector_centrality(G, max_iter=500)
        stkdd_dict = {n: compute_STKDD(disorder_dict[n], eig[n], 0) for n in G.nodes()}
        st.session_state['stkdd_dict'] = stkdd_dict
        flux_dict = compute_DFWMIN(G, disorder_dict, stkdd_dict)
        st.session_state['flux_dict'] = flux_dict
        pos = nx.spring_layout(G, seed=42)
        st.session_state['pos'] = pos

        fig = plot_network(G, pos, stkdd_dict, metric_name="STKDD")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Tab 4: Network Metrics ----------------------------
with tabs[3]:
    st.header("Network Metrics")
    if 'G' in st.session_state:
        G = st.session_state['G']
        metrics = pd.DataFrame({
            'Protein': list(G.nodes()),
            'Degree': [G.degree(n) for n in G.nodes()],
            'Betweenness': [nx.betweenness_centrality(G)[n] for n in G.nodes()],
            'Closeness': [nx.closeness_centrality(G)[n] for n in G.nodes()],
            'Eigenvector': [nx.eigenvector_centrality(G, max_iter=500)[n] for n in G.nodes()]
        })
        st.session_state['metrics'] = metrics
        st.dataframe(metrics)

# ---------------------------- Tab 5: Protein Details ----------------------------
with tabs[4]:
    st.header("Protein Details from UniProt")
    if 'seq_dict' in st.session_state:
        for protein in st.session_state['seq_dict'].keys():
            info = fetch_uniprot_info(protein)
            if info:
                st.subheader(protein)
                st.json(info)

# ---------------------------- Tab 6: Sequences / FASTA ----------------------------
with tabs[5]:
    st.header("FASTA Sequences")
    if 'seq_dict' in st.session_state:
        st.text(st.session_state['seq_dict'])

# ---------------------------- Tab 7: Motifs / Domains ----------------------------
with tabs[6]:
    st.header("Motifs & Domains")
    if 'seq_dict' in st.session_state:
        for protein in st.session_state['seq_dict']:
            info = fetch_uniprot_info(protein)
            if info:
                features = info.get('features', [])
                st.json(features)

# ---------------------------- Tab 8: 3D Viewer ----------------------------
with tabs[7]:
    st.header("3D AlphaFold Viewer")
    if 'seq_dict' in st.session_state:
        for protein in st.session_state['seq_dict']:
            pdb = fetch_alphafold_structure(protein)
            if pdb:
                st.subheader(protein)
                view = py3Dmol.view(width=800,height=400)
                view.addModel(pdb,'pdb')
                view.setStyle({'cartoon': {'color':'spectrum'}})
                view.zoomTo()
                html_view = view._make_html()
                st.components.v1.html(html_view, height=400, width=800)

# ---------------------------- Tab 9: Disorder & Stability ----------------------------
with tabs[8]:
    st.header("Intrinsic Disorder & Stability")
    if 'seq_dict' in st.session_state:
        for protein, seq in st.session_state['seq_dict'].items():
            disorder_scores = compute_disorder(seq)
            st.subheader(protein)
            st.line_chart(disorder_scores)

# ---------------------------- Tab 10: DMNFZ Explorer ----------------------------
with tabs[9]:
    st.header("DMNFZ Explorer")
    if 'G' in st.session_state and 'disorder_dict' in st.session_state:
        dmnfz_dict = compute_DMNFZ(st.session_state['G'], st.session_state['disorder_dict'])
        st.json(dmnfz_dict)

# ---------------------------- Tab 11: DFWMIN Explorer ----------------------------
with tabs[10]:
    st.header("DFWMIN Explorer")
    if 'flux_dict' in st.session_state:
        st.json(st.session_state['flux_dict'])

# ---------------------------- Tab 12: Centrality Measures ----------------------------
with tabs[11]:
    st.header("Advanced Centrality Measures")
    if 'G' in st.session_state:
        G = st.session_state['G']
        degree_c = dict(G.degree())
        betweenness_c = nx.betweenness_centrality(G)
        closeness_c = nx.closeness_centrality(G)
        eigen_c = nx.eigenvector_centrality(G, max_iter=500)
        centrality_df = pd.DataFrame({
            'Protein': list(G.nodes()),
            'Degree': [degree_c[n] for n in G.nodes()],
            'Betweenness': [betweenness_c[n] for n in G.nodes()],
            'Closeness': [closeness_c[n] for n in G.nodes()],
            'Eigenvector': [eigen_c[n] for n in G.nodes()]
        })
        st.dataframe(centrality_df)

# ---------------------------- Tab 13: Community Detection ----------------------------
with tabs[12]:
    st.header("Community Detection")
    if 'G' in st.session_state:
        G = st.session_state['G']
        communities_gen = community.label_propagation_communities(G)
        comm_dict = {f'Community {i+1}': list(c) for i,c in enumerate(communities_gen)}
        st.json(comm_dict)

# ---------------------------- Tab 14: Network Robustness ----------------------------
with tabs[13]:
    st.header("Network Robustness Simulation")
    if 'G' in st.session_state:
        G = st.session_state['G']
        pos = st.session_state['pos']
        degrees = dict(G.degree())
        top_node = max(degrees, key=degrees.get)
        G_temp = G.copy()
        G_temp.remove_node(top_node)
        fig = plot_network(G_temp, pos, st.session_state['stkdd_dict'], metric_name="STKDD", highlight_nodes=[top_node])
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Tab 15: Seq-Structure Mapping ----------------------------
with tabs[14]:
    st.header("Sequence-Structure Mapping")
    if 'seq_dict' in st.session_state:
        for protein, seq in st.session_state['seq_dict'].items():
            disorder_scores = compute_disorder(seq)
            st.subheader(f"{protein} Disorder vs Residue")
            st.line_chart(disorder_scores)

# ---------------------------- Tab 16: Downloads / Exports ----------------------------
with tabs[15]:
    st.header("Download Metrics & Graph Data")
    if 'metrics' in st.session_state:
        csv = st.session_state['metrics'].to_csv(index=False).encode()
        st.download_button(label="Download Metrics CSV", data=csv, file_name='metrics.csv', mime='text/csv')
    if 'df_edges' in st.session_state:
        csv_edges = st.session_state['df_edges'].to_csv(index=False).encode()
        st.download_button(label="Download Edge List CSV", data=csv_edges, file_name='edges.csv', mime='text/csv')
    if 'G' in st.session_state:
        nx.write_graphml(st.session_state['G'], "network.graphml")
        with open("network.graphml", "r") as f:
            st.download_button(label="Download GraphML", data=f, file_name="network.graphml")
