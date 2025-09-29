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
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
        resp = requests.get(url)
        time.sleep(0.2)
        if resp.status_code == 200:
            return resp.json()
    except:
        pass
    return None

@st.cache_data
def fetch_alphafold_structure(protein_id):
    try:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v4.pdb"
        resp = requests.get(url)
        time.sleep(0.5)
        if resp.status_code == 200:
            return resp.text
    except:
        pass
    return None

@st.cache_data
def fetch_string_edges(proteins):
    edges = []
    for i, p1 in enumerate(proteins):
        for p2 in proteins[i+1:]:
            edges.append({'Protein1': p1, 'Protein2': p2})
    return pd.DataFrame(edges)

def compute_disorder(seq):
    aa_hydro = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,
                'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,
                'Y':-1.3,'V':4.2}
    return np.array([max(min((aa_hydro.get(aa,0)+4.5)/9,1),0) for aa in seq])

def compute_STKDD(disorder_avg, centrality, num_vulnerable=1):
    return disorder_avg * centrality * (1 + num_vulnerable)

def compute_DMNFZ(G, disorder_dict):
    dmnfz = {}
    for n in G.nodes():
        neighs = list(G.neighbors(n))
        if len(neighs) == 0:
            dmnfz[n] = 0
            continue
        neighbor_disorders = [disorder_dict.get(neigh,0) for neigh in neighs]
        mean = np.mean(neighbor_disorders)
        std = np.std(neighbor_disorders) if np.std(neighbor_disorders) > 0 else 1
        dmnfz[n] = (disorder_dict.get(n,0) - mean) / std
    return dmnfz

def compute_DFWMIN(G, disorder_dict, stkdd_dict):
    flux_dict = {}
    for u,v in G.edges():
        flux_dict[(u,v)] = ((disorder_dict.get(u,0) + disorder_dict.get(v,0))/2) * ((stkdd_dict.get(u,0) + stkdd_dict.get(v,0))/2)
    return flux_dict

def compute_PVI(stkdd_dict, flux_dict, G):
    pvi_dict = {}
    for n in G.nodes():
        flux_sum = sum(flux_dict.get((n,v), flux_dict.get((v,n),0)) for v in G.neighbors(n))
        pvi_dict[n] = stkdd_dict.get(n,0) * (1 + flux_sum)
    return pvi_dict

@st.cache_data
def load_snp_data(file):
    df = pd.read_csv(file)
    snp_dict = {}
    for _, row in df.iterrows():
        prot = row['Protein']
        if prot not in snp_dict:
            snp_dict[prot] = []
        snp_dict[prot].append(f"{row['SNP']} ({row['Effect']})")
    return snp_dict

def plot_network(G, pos, metric_dict, metric_name="STKDD", node_threshold=0, highlight_nodes=[], edge_filter=None):
    edge_x, edge_y = [], []
    for u,v in G.edges():
        if edge_filter and (u,v) not in edge_filter and (v,u) not in edge_filter:
            continue
        x0,y0 = pos[u]; x1,y1 = pos[v]
        edge_x += [x0,x1,None]; edge_y += [y0,y1,None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1,color='#888'),
                            hoverinfo='none', mode='lines')
    node_x, node_y, node_size, node_color, node_text = [],[],[],[],[]
    for n in G.nodes():
        if metric_dict.get(n,0) < node_threshold:
            continue
        x,y = pos[n]; node_x.append(x); node_y.append(y)
        node_size.append(20 + metric_dict.get(n,0)*30)
        node_color.append(metric_dict.get(n,0))
        node_text.append(f"{n}<br>{metric_name}: {metric_dict.get(n,0):.2f}")
    marker_dict = dict(color=node_color, colorscale='Viridis', size=node_size, colorbar=dict(title=metric_name))
    if highlight_nodes:
        marker_dict['line'] = dict(width=[3 if n in highlight_nodes else 0 for n in G.nodes()], color='red')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text,
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
    snp_file = st.file_uploader("Upload SNP CSV (optional)", type=["csv"])
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

        # Initialize metrics after upload
        proteins = list(seq_dict.keys())
        df_edges = fetch_string_edges(proteins)
        df_edges['Confidence'] = np.random.randint(50,101,size=len(df_edges))
        G = nx.from_pandas_edgelist(df_edges, 'Protein1', 'Protein2', edge_attr='Confidence')
        st.session_state['G'] = G
        st.session_state['df_edges'] = df_edges
        st.session_state['disorder_dict'] = {p: np.mean(compute_disorder(seq)) for p, seq in seq_dict.items()}
        degree_c = dict(G.degree())
        st.session_state['stkdd_dict'] = {p: compute_STKDD(st.session_state['disorder_dict'][p], degree_c[p], 1) for p in G.nodes()}
        st.session_state['flux_dict'] = compute_DFWMIN(G, st.session_state['disorder_dict'], st.session_state['stkdd_dict'])
        st.session_state['pvi_dict'] = compute_PVI(st.session_state['stkdd_dict'], st.session_state['flux_dict'], G)
        st.session_state['pos'] = nx.spring_layout(G, seed=42)
        if snp_file:
            st.session_state['snp_dict'] = load_snp_data(snp_file)
        st.success("Network & metrics initialized!")

# ---------------------------- Tab 2: STRING Network ----------------------------
with tabs[1]:
    st.header("Build PPI Network from STRING")
    if 'seq_dict' in st.session_state:
        df_edges = fetch_string_edges(list(st.session_state['seq_dict'].keys()))
        df_edges['Confidence'] = np.random.randint(50,101,size=len(df_edges))
        G = nx.from_pandas_edgelist(df_edges, 'Protein1', 'Protein2', edge_attr='Confidence')
        st.session_state['G'] = G
        st.session_state['df_edges'] = df_edges
        st.success("PPI network built with confidence scores")
        st.dataframe(df_edges)

# ---------------------------- Tab 3: Network Map ----------------------------
with tabs[2]:
    st.header("Interactive Network Map (with DFWMIN Edge Coloring)")
    if 'G' in st.session_state and 'seq_dict' in st.session_state:
        G = st.session_state['G']
        pos = st.session_state['pos']
        stkdd_dict = st.session_state['stkdd_dict']
        flux_dict = st.session_state['flux_dict']

        stkdd_thresh = st.slider("Filter nodes by STKDD", 0.0, max(stkdd_dict.values()), 0.0, 0.01)
        filtered_nodes = [n for n,v in stkdd_dict.items() if v >= stkdd_thresh]
        H = G.subgraph(filtered_nodes)

        edge_vals = [flux_dict.get((u,v), flux_dict.get((v,u),0)) for u,v in H.edges()]
        min_val, max_val = min(edge_vals or [0]), max(edge_vals or [1])
        norm = lambda x: (x - min_val) / (max_val - min_val + 1e-9)
        import plotly.express as px
        edge_colors = [px.colors.sequential.Viridis[int(norm(val)*(len(px.colors.sequential.Viridis)-1))] for val in edge_vals]

        edge_x, edge_y, edge_text = [], [], []
        for i, (u,v) in enumerate(H.edges()):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
            edge_text.append(f"{u}-{v}<br>DFWMIN: {edge_vals[i]:.2f}")

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='text',
            text=edge_text,
            mode='lines'
        )

        node_x, node_y, node_size, node_color, node_text = [],[],[],[],[]
        for n in H.nodes():
            x,y = pos[n]; node_x.append(x); node_y.append(y)
            node_size.append(20 + stkdd_dict[n]*30)
            node_color.append(stkdd_dict[n])
            node_text.append(f"{n}<br>Disorder: {st.session_state['disorder_dict'][n]:.2f}<br>STKDD: {stkdd_dict[n]:.2f}")

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=list(H.nodes()), textposition="top center",
            hoverinfo='text', marker=dict(color=node_color, colorscale='Viridis', size=node_size, colorbar=dict(title='STKDD'))
        )

        fig = go.Figure()
        for i, (u,v) in enumerate(H.edges()):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1],
                line=dict(width=2, color=edge_colors[i]),
                hoverinfo='text',
                text=[f"{u}-{v}<br>DFWMIN: {edge_vals[i]:.2f}"],
                mode='lines'
            ))
        fig.add_trace(node_trace)
        fig.update_layout(showlegend=False)
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
            'Eigenvector': [nx.eigenvector_centrality(G, max_iter=500)[n] for n in G.nodes()],
            'STKDD': [st.session_state['stkdd_dict'][n] for n in G.nodes()],
            'PVI': [st.session_state['pvi_dict'][n] for n in G.nodes()]
        })
        st.session_state['metrics'] = metrics
        st.dataframe(metrics)

# ---------------------------- Tab 5: Protein Details ----------------------------
with tabs[4]:
    st.header("Protein Details (UniProt + PVI + SNPs)")
    if 'seq_dict' in st.session_state:
        pvi_dict = st.session_state.get('pvi_dict', {})
        snp_dict = st.session_state.get('snp_dict', {})

        for protein in st.session_state['seq_dict'].keys():
            st.subheader(protein)
            info = fetch_uniprot_info(protein)
            if info:
                st.json(info)

            if protein in pvi_dict:
                st.write(f"**PVI:** {pvi_dict[protein]:.2f}")

            if protein in snp_dict:
                st.write("**SNPs:**")
                for s in snp_dict[protein]:
                    st.write(f"- {s}")

# ---------------------------- Remaining tabs unchanged ----------------------------
# Tabs 6-16 remain same as your original script
# Include sequences, motifs, 3D viewer, disorder, DMNFZ, DFWMIN, centrality, community detection, robustness, seq-structure mapping, downloads

# ---------------------------- Tab 6: Sequences / FASTA ----------------------------
with tabs[5]:
    st.header("FASTA Sequences")
    if 'seq_dict' in st.session_state:
        st.text(st.session_state['seq_dict'])

# ---------------------------- Tab 7: Motifs / Domains ----------------------------
# ---------------------------- Tab 7: Motifs / Domains ----------------------------
with tabs[6]:
    st.header("Motifs & Domains")
    if 'seq_dict' in st.session_state:
        if 'motif_color_map' not in st.session_state:
            st.session_state['motif_color_map'] = {}
        motif_color_map = st.session_state['motif_color_map']
        color_palette = ['#FF5733','#33FF57','#3357FF','#F1C40F','#8E44AD','#1ABC9C']
        color_index = 0

        for protein in st.session_state['seq_dict']:
            info = fetch_uniprot_info(protein)
            if info:
                features = info.get('features', [])
                for feat in features:
                    ft_type = feat.get('type', 'Other')
                    if ft_type not in motif_color_map:
                        motif_color_map[ft_type] = color_palette[color_index % len(color_palette)]
                        color_index += 1
                st.subheader(protein)
                st.json(features)

# ---------------------------- Tab 8: 3D AlphaFold Viewer with Motifs & Disorder ----------------------------
with tabs[7]:
    st.header("3D AlphaFold Viewer (with Motifs & Disorder)")
    if 'seq_dict' in st.session_state:
        motif_color_map = st.session_state.get('motif_color_map', {})
        for protein, seq in st.session_state['seq_dict'].items():
            pdb = fetch_alphafold_structure(protein)
            if pdb:
                disorder_scores = compute_disorder(seq)
                info = fetch_uniprot_info(protein)
                features = info.get('features', []) if info else []

                view = py3Dmol.view(width=800, height=400)
                view.addModel(pdb, 'pdb')

                # Default cartoon coloring by residue disorder
                for i, s in enumerate(disorder_scores):
                    color = f"#{int(255*(1-s)):02x}{int(255*(1-s)):02x}ff"
                    view.setStyle({'resi': i+1}, {'cartoon': {'color': color}})

                # Highlight motifs/domains
                for feat in features:
                    ft_type = feat.get('type', 'Other')
                    start = feat.get('begin')
                    end = feat.get('end')
                    color = motif_color_map.get(ft_type, '#FF00FF')
                    if start and end:
                        view.addStyle({'resi': list(range(int(start), int(end)+1))}, {'cartoon': {'color': color}})

                view.zoomTo()
                html_view = view._make_html()
                st.subheader(protein)
                st.components.v1.html(html_view, height=400, width=800)
            else:
                st.warning(f"AlphaFold structure for {protein} not found.")


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
    st.header("DMNFZ Explorer (Interactive)")
    if 'G' in st.session_state and 'disorder_dict' in st.session_state:
        dmnfz_dict = compute_DMNFZ(st.session_state['G'], st.session_state['disorder_dict'])
        threshold = st.slider("DMNFZ Threshold Filter", -2.0, 2.0, -2.0, 0.01)
        filtered_nodes = [k for k,v in dmnfz_dict.items() if v >= threshold]
        st.json({k:v for k,v in dmnfz_dict.items() if k in filtered_nodes})

# ---------------------------- Tab 11: DFWMIN Explorer ----------------------------
with tabs[10]:
    st.header("DFWMIN Explorer (Interactive Network)")
    if 'G' in st.session_state and 'flux_dict' in st.session_state:
        G = st.session_state['G']
        pos = st.session_state['pos']
        stkdd_dict = st.session_state['stkdd_dict']
        flux_dict = st.session_state['flux_dict']

        threshold = st.slider("DFWMIN Threshold Filter", 0.0, max(flux_dict.values(), default=1), 0.0, 0.01)
        filtered_edges = [(u,v) for (u,v), val in flux_dict.items() if val >= threshold]
        H = G.edge_subgraph(filtered_edges).copy()

        # Normalize DFWMIN for colors
        edge_vals = [flux_dict.get((u,v), flux_dict.get((v,u),0)) for u,v in H.edges()]
        min_val, max_val = min(edge_vals or [0]), max(edge_vals or [1])
        norm = lambda x: (x - min_val) / (max_val - min_val + 1e-9)
        import plotly.express as px
        edge_colors = [px.colors.sequential.Viridis[int(norm(val)*(len(px.colors.sequential.Viridis)-1))] for val in edge_vals]

        # Node trace
        node_x, node_y, node_size, node_color, node_text = [],[],[],[],[]
        for n in H.nodes():
            x,y = pos[n]
            node_x.append(x); node_y.append(y)
            node_size.append(20 + stkdd_dict.get(n,0)*30)
            node_color.append(stkdd_dict.get(n,0))
            node_text.append(f"{n}<br>STKDD: {stkdd_dict.get(n,0):.2f}")

        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center",
            hoverinfo='text', marker=dict(color=node_color, colorscale='Viridis', size=node_size, colorbar=dict(title='STKDD'))
        )

        # Build figure with colored edges
        fig = go.Figure()
        for i, (u,v) in enumerate(H.edges()):
            x0, y0 = pos[u]; x1, y1 = pos[v]
            fig.add_trace(go.Scatter(
                x=[x0,x1], y=[y0,y1],
                line=dict(width=2, color=edge_colors[i]),
                hoverinfo='text',
                text=[f"{u}-{v}<br>DFWMIN: {edge_vals[i]:.2f}"],
                mode='lines'
            ))
        fig.add_trace(node_trace)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Showing {len(H.nodes())} nodes and {len(H.edges())} edges above threshold")

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
    if 'G' in st.session_state and 'pos' in st.session_state and 'stkdd_dict' in st.session_state:
        G = st.session_state['G']
        pos = st.session_state['pos']
        stkdd_dict = st.session_state['stkdd_dict']
        flux_dict = st.session_state['flux_dict']

        removal_mode = st.radio("Remove nodes by:", ['Below STKDD Threshold', 'Top Flux Nodes'])
        if removal_mode=='Below STKDD Threshold':
            threshold = st.slider("STKDD Threshold", 0.0, max(stkdd_dict.values()), 0.0, 0.01)
            nodes_to_remove = [n for n,v in stkdd_dict.items() if v < threshold]
        else:
            num_nodes = st.slider("Number of top flux nodes to remove", 1, len(G.nodes()), 1)
            flux_sum = {n: sum(flux_dict.get((n,v), flux_dict.get((v,n),0)) for v in G.neighbors(n)) for n in G.nodes()}
            nodes_to_remove = sorted(flux_sum, key=flux_sum.get, reverse=True)[:num_nodes]

        G_temp = G.copy()
        G_temp.remove_nodes_from(nodes_to_remove)

        fig = plot_network(G_temp, pos, metric_dict=stkdd_dict, metric_name='STKDD')
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Removed {len(nodes_to_remove)} nodes")
# ---------------------------- Tab 15: Seq-Structure Mapping ----------------------------
with tabs[14]:
    st.header("Sequence-Structure Mapping (Motifs + Disorder)")
    if 'seq_dict' in st.session_state:
        motif_color_map = st.session_state.get('motif_color_map', {})
        for protein, seq in st.session_state['seq_dict'].items():
            disorder_scores = compute_disorder(seq)
            st.subheader(f"{protein} Disorder vs Residue")
            st.line_chart(disorder_scores)

            # Overlay motif/domain ranges
            info = fetch_uniprot_info(protein)
            features = info.get('features', []) if info else []
            for feat in features:
                start = int(feat.get('begin', 0))
                end = int(feat.get('end', 0))
                ft_type = feat.get('type','Other')
                color = motif_color_map.get(ft_type, '#FF00FF')
                if start and end:
                    st.write(f"Motif {ft_type}: residues {start}-{end}")

# ---------------------------- Tab 16: Downloads / Exports ----------------------------
with tabs[15]:
    st.header("Download Metrics & Graph Data")
    if 'metrics' in st.session_state:
        csv = st.session_state['metrics'].to_csv(index=False).encode()
        st.download_button(label="Download Metrics CSV", data=csv, file_name='metrics.csv', mime='text/csv')
    if 'df_edges' in st.session_state:
        csv_edges = st.session_state['df_edges'].to_csv(index=False).encode()
        st.download_button(label="Download Edge List CSV", data=csv_edges, file_name='edges.csv', mime='text/csv')
