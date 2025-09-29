import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import py3Dmol
import requests
from networkx.algorithms import community

st.set_page_config(page_title="IPF Disorder-Weighted Protein Network Explorer", layout="wide")

# ---------------------------- Tabs ----------------------------
tabs = st.tabs([
    "Upload / Files", "Network Map", "Network Metrics", "Protein Details",
    "Sequences / FASTA", "Motifs / Domains", "3D Viewer", "Disorder & Stability",
    "DMNFZ Explorer", "DFWMIN Explorer", "Centrality Measures", "Community Detection",
    "Network Robustness", "Seq-Structure Mapping", "Downloads / Exports"
])

# ---------------------------- Helper Functions ----------------------------
def build_graph(df_edges):
    return nx.from_pandas_edgelist(df_edges, 'Protein1', 'Protein2')

def compute_disorder(seq):
    aa_hydro = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
                'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,
                'Y':-1.3,'V':4.2}
    scores = [max(min((aa_hydro.get(aa,0)+4.5)/9,1),0) for aa in seq]
    return np.array(scores)

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

def fetch_uniprot_motifs(protein_id):
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    resp = requests.get(url)
    if resp.status_code==200:
        features = resp.json().get('features',[])
        return [f for f in features if f['type'] in ['DOMAIN','MOTIF']]
    return []

def fetch_alphafold_structure(protein_id):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v4.pdb"
    r = requests.get(url)
    if r.status_code==200:
        return r.text
    return None

# ---------------------------- Tab 1: Upload ----------------------------
with tabs[0]:
    st.header("Upload Files")
    uploaded_csv = st.file_uploader("Upload network CSV (edges)", type=["csv"])
    uploaded_fasta = st.file_uploader("Upload FASTA sequences", type=["fasta"])
    uploaded_snp = st.file_uploader("Optional SNP/PTM CSV", type=["csv"])

    if uploaded_csv:
        df_edges = pd.read_csv(uploaded_csv)
        st.session_state['df_edges'] = df_edges
        st.success("Network CSV loaded")
        st.dataframe(df_edges)

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

    if uploaded_snp:
        df_snp = pd.read_csv(uploaded_snp)
        st.session_state['df_snp'] = df_snp
        st.success("SNP/PTM data loaded")

# ---------------------------- Tab 2: Network Map ----------------------------
with tabs[1]:
    st.header("Interactive Network Map")
    if 'df_edges' in st.session_state:
        G = build_graph(st.session_state['df_edges'])
        st.session_state['G'] = G

        disorder_dict = {}
        for n in G.nodes():
            seq = st.session_state.get('seq_dict', {}).get(n,'')
            disorder_dict[n] = np.mean(compute_disorder(seq)) if seq else np.random.rand()
        st.session_state['disorder_dict'] = disorder_dict

        eig = nx.eigenvector_centrality(G, max_iter=500)
        stkdd_dict = {n: compute_STKDD(disorder_dict[n], eig[n], 0) for n in G.nodes()}
        st.session_state['stkdd_dict'] = stkdd_dict

        flux_dict = compute_DFWMIN(G, disorder_dict, stkdd_dict)
        st.session_state['flux_dict'] = flux_dict
        pos = nx.spring_layout(G, seed=42)
        st.session_state['pos'] = pos

        # Plotly network visualization
        edge_x, edge_y = [], []
        for u,v in G.edges():
            x0,y0 = pos[u]; x1,y1 = pos[v]
            edge_x += [x0,x1,None]; edge_y += [y0,y1,None]

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1,color='#888'),
                                hoverinfo='text', mode='lines')

        node_x, node_y, node_size, node_color, node_text = [],[],[],[],[]
        for n in G.nodes():
            x,y = pos[n]; node_x.append(x); node_y.append(y)
            node_size.append(20 + stkdd_dict[n]*30)
            node_color.append(stkdd_dict[n])
            seq_len = len(st.session_state.get('seq_dict', {}).get(n,'')) or 1
            snp_positions = st.session_state.get('df_snp', pd.DataFrame()).query(f"Protein=='{n}'")['Position'].tolist() if 'df_snp' in st.session_state else []
            vulnerable = compute_vulnerable_regions(compute_disorder(st.session_state.get('seq_dict', {}).get(n,'')), snp_positions)
            node_text.append(f"{n}<br>Disorder: {disorder_dict[n]:.2f}<br>Vulnerable: {vulnerable}<br>SNPs: {snp_positions}")

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), textposition="top center",
                                hoverinfo='text', marker=dict(color=node_color, colorscale='Viridis', size=node_size,
                                                              colorbar=dict(title="STKDD")))

        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------- Tab 3: Network Metrics ----------------------------
with tabs[2]:
    st.header("Network Metrics")
    if 'G' in st.session_state:
        G = st.session_state['G']
        metrics = pd.DataFrame({
            'Protein': list(G.nodes()),
            'Degree': [G.degree(n) for n in G.nodes()],
            'Betweenness': [nx.betweenness_centrality(G)[n] for n in G.nodes()],
            'Closeness': [nx.closeness_centrality(G)[n] for n in G.nodes()],
            'Eigenvector': [nx.eigenvector_centrality(G, max_iter=500)[n] for n in G.nodes()],
            'STKDD': [st.session_state['stkdd_dict'][n] for n in G.nodes()]
        })
        st.session_state['metrics'] = metrics
        st.dataframe(metrics)

# ---------------------------- Tabs 4–15: Full Implementation ----------------------------
# Tabs 4: Protein Details
with tabs[3]:
    st.header("Protein Details")
    protein_id = st.text_input("Enter UniProt ID")
    if protein_id:
        motifs = fetch_uniprot_motifs(protein_id)
        st.json(motifs)
        disorder_score = st.session_state.get('disorder_dict', {}).get(protein_id, None)
        st.write(f"Average Disorder Score: {disorder_score}")

# Tabs 5: Sequences / FASTA
with tabs[4]:
    st.header("Sequences / FASTA")
    if 'seq_dict' in st.session_state:
        st.dataframe(pd.DataFrame(list(st.session_state['seq_dict'].items()), columns=['Protein','Sequence']))

# Tabs 6: Motifs / Domains
with tabs[5]:
    st.header("Motifs / Domains")
    if protein_id and 'motifs' in locals():
        for f in motifs:
            st.write(f"{f['type']}: {f.get('description','')} ({f.get('begin','')} - {f.get('end','')})")

# Tabs 7: 3D AlphaFold Viewer
with tabs[6]:
    st.header("3D AlphaFold Viewer")
    if protein_id:
        pdb_data = fetch_alphafold_structure(protein_id)
        if pdb_data:
            view = py3Dmol.view(width=800, height=600)
            view.addModel(pdb_data, 'pdb')
            view.setStyle({'cartoon': {'color':'spectrum'}})
            view.zoomTo()
            view.show()
            st.write(view)
        else:
            st.warning("AlphaFold structure not found.")

# Tabs 8: Disorder & Stability
with tabs[7]:
    st.header("Disorder & Stability")
    if 'seq_dict' in st.session_state:
        fig, ax = plt.subplots(figsize=(10,4))
        for prot, seq in st.session_state['seq_dict'].items():
            d_scores = compute_disorder(seq)
            ax.plot(range(len(seq)), d_scores, label=prot)
        ax.set_title("Disorder Profiles")
        ax.set_xlabel("Residue")
        ax.set_ylabel("Disorder Score")
        ax.legend()
        st.pyplot(fig)

# Tabs 9: DMNFZ Explorer
with tabs[8]:
    st.header("DMNFZ Explorer")
    if 'G' in st.session_state:
        dmnfz = compute_DMNFZ(st.session_state['G'], st.session_state['disorder_dict'])
        st.dataframe(pd.DataFrame(list(dmnfz.items()), columns=['Protein','DMNFZ']))

# Tabs 10: DFWMIN Explorer
with tabs[9]:
    st.header("DFWMIN Explorer")
    if 'flux_dict' in st.session_state:
        flux_df = pd.DataFrame(list(st.session_state['flux_dict'].items()), columns=['Edge','Flux'])
        st.dataframe(flux_df)

# Tabs 11: Centrality Measures
with tabs[10]:
    st.header("Centrality Measures")
    if 'G' in st.session_state:
        centrality_df = pd.DataFrame({
            'Protein': list(st.session_state['G'].nodes()),
            'Degree': [st.session_state['G'].degree(n) for n in st.session_state['G'].nodes()],
            'Betweenness': [nx.betweenness_centrality(st.session_state['G'])[n] for n in st.session_state['G'].nodes()],
            'Closeness': [nx.closeness_centrality(st.session_state['G'])[n] for n in st.session_state['G'].nodes()],
            'Eigenvector': [nx.eigenvector_centrality(st.session_state['G'], max_iter=500)[n] for n in st.session_state['G'].nodes()]
        })
        st.dataframe(centrality_df)

# Tabs 12: Community Detection
with tabs[11]:
    st.header("Community Detection / Clustering")
    if 'G' in st.session_state:
        comms = community.label_propagation_communities(st.session_state['G'])
        comm_dict = {i+1:list(c) for i,c in enumerate(comms)}
        st.json(comm_dict)

# Tabs 13: Network Robustness
with tabs[12]:
    st.header("Network Robustness")
    if 'G' in st.session_state:
        G_temp = st.session_state['G'].copy()
        degs = dict(G_temp.degree())
        top_node = max(degs, key=degs.get)
        G_temp.remove_node(top_node)
        pos = nx.spring_layout(G_temp, seed=42)
        nx.draw(G_temp, pos, with_labels=True, node_color='red', edge_color='gray', node_size=1200)
        st.pyplot(plt)

# Tabs 14: Sequence-Structure Mapping
with tabs[13]:
    st.header("Sequence-Structure Mapping")
    st.write("Mapping disorder scores onto network nodes for visualization...")
    if 'disorder_dict' in st.session_state:
        mapping_df = pd.DataFrame(list(st.session_state['disorder_dict'].items()), columns=['Protein','Disorder'])
        st.dataframe(mapping_df)

# Tabs 15: Downloads / Exports
with tabs[14]:
    st.header("Download Metrics & Graphs")
    if 'metrics' in st.session_state:
        csv = st.session_state['metrics'].to_csv(index=False).encode()
        st.download_button("Download Metrics CSV", csv, "metrics.csv", "text/csv")
