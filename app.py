import streamlit as st
import requests
import py3Dmol
import matplotlib.pyplot as plt
from collections import deque
import itertools
import pandas as pd

# -----------------------------
# Page configuration
st.set_page_config(
    page_title="Protein Network Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Header
st.markdown("<h1 style='text-align: center; color: #4B0082;'>🧬 Protein Network Explorer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:16px; color: #6A5ACD;'>Explore human protein networks, sequences, motifs, and 3D structures!</p>", unsafe_allow_html=True)

# -----------------------------
# Sidebar for file upload
st.sidebar.header("Upload Protein Network CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file:
    data = list(pd.read_csv(uploaded_file).to_dict('records'))

    # Filter human proteins
    human_data = [row for row in data if str(row.get("TaxID", "9606")) == "9606"]
    st.sidebar.success(f"Filtered human proteins: {len(human_data)}")

    # Build adjacency dictionary
    adj = {}
    proteins = set()
    for row in human_data:
        p1, p2 = row["Protein1"], row["Protein2"]
        proteins.update([p1, p2])
        adj.setdefault(p1, set()).add(p2)
        adj.setdefault(p2, set()).add(p1)

    st.sidebar.info(f"Network built! Number of proteins: {len(proteins)}")

    # -----------------------------
    # Functions for network metrics
    def bfs_shortest_paths(graph, start):
        visited = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    visited[neighbor] = visited[node] + 1
                    queue.append(neighbor)
        return visited

    # Closeness Centrality
    closeness = {node: (len(bfs_shortest_paths(adj, node))-1)/sum(bfs_shortest_paths(adj, node).values()) if len(adj[node])>0 else 0 for node in proteins}

    # Betweenness Centrality
    betweenness = dict.fromkeys(proteins, 0)
    for s in proteins:
        sp_count = {v:0 for v in proteins}
        sp_pred = {v:[] for v in proteins}
        D = {v:-1 for v in proteins}
        D[s] = 0
        Q = deque([s])
        while Q:
            v = Q.popleft()
            for w in adj[v]:
                if D[w] < 0:
                    Q.append(w)
                    D[w] = D[v]+1
                if D[w] == D[v]+1:
                    sp_pred[w].append(v)
                    sp_count[w] += 1
        delta = dict.fromkeys(proteins, 0)
        nodes_sorted = sorted(D.items(), key=lambda x: -x[1])
        for w, _ in nodes_sorted:
            for v in sp_pred[w]:
                delta[v] += (1 + delta[w])
            if w != s:
                betweenness[w] += delta[w]

    # Clustering Coefficient
    clustering = {}
    for node in proteins:
        neighbors = adj[node]
        if len(neighbors) < 2:
            clustering[node] = 0
        else:
            links = 0
            for u,v in itertools.combinations(neighbors, 2):
                if u in adj[v]:
                    links += 1
            clustering[node] = 2*links/(len(neighbors)*(len(neighbors)-1))

    # -----------------------------
    # Tabs
    tabs = st.tabs(["📊 Network Metrics", "🧩 Protein Details", "🧬 3D Structure Viewer", "🌐 Interactive Network"])

    # ----- Tab 1: Network Metrics -----
    with tabs[0]:
        st.subheader("Network Metrics for Each Protein")
        metrics_df = pd.DataFrame({
            "Protein": list(proteins),
            "Closeness": [closeness[p] for p in proteins],
            "Betweenness": [betweenness[p] for p in proteins],
            "Clustering": [clustering[p] for p in proteins]
        })

        # Colorful highlighting
        def color_metrics(val):
            if isinstance(val, float):
                if val > metrics_df["Closeness"].quantile(0.75):
                    return 'background-color: #FFD700; color: black'  # gold
                elif val > metrics_df["Betweenness"].quantile(0.75):
                    return 'background-color: #ADFF2F; color: black'  # green
                elif val > metrics_df["Clustering"].quantile(0.75):
                    return 'background-color: #FF69B4; color: black'  # pink
            return ''
        st.dataframe(metrics_df.style.applymap(color_metrics, subset=["Closeness","Betweenness","Clustering"]), height=400)

    # ----- Tab 2: Protein Details -----
    with tabs[1]:
        st.subheader("Protein Sequence & Motifs/Domains")
        selected_protein = st.selectbox("Select a protein", list(proteins))
        if selected_protein:
            col1, col2 = st.columns(2)
            with col1:
                uni_url = f"https://rest.uniprot.org/uniprotkb/{selected_protein}.fasta"
                r = requests.get(uni_url)
                if r.ok:
                    seq = "".join(r.text.split("\n")[1:])
                    st.text_area("Protein Sequence (FASTA)", seq, height=300)
                else:
                    st.warning("Sequence not found in UniProt.")

            with col2:
                ps_url = f"https://www.ebi.ac.uk/proteins/api/proteins/{selected_protein}"
                headers = {"Accept": "application/json"}
                r2 = requests.get(ps_url, headers=headers)
                if r2.ok:
                    info = r2.json()
                    features = info.get("features", [])
                    motifs = [f"{f['type']}: {f['description']}" for f in features]
                    st.markdown("<h4 style='color:#4B0082;'>Motifs/Domains</h4>", unsafe_allow_html=True)
                    for m in motifs:
                        st.markdown(f"<p style='background-color:#E6E6FA; padding:5px; border-radius:5px;'>{m}</p>", unsafe_allow_html=True)
                else:
                    st.warning("Motifs/Domains not found.")

    # ----- Tab 3: 3D Structure Viewer -----
    with tabs[2]:
        st.subheader("AlphaFold 3D Structure Viewer")
        structure_id = st.text_input("Enter AlphaFold ID (e.g., P53_HUMAN)")
        if structure_id:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{structure_id}-F1-model_v4.pdb"
            r3 = requests.get(url)
            if r3.ok:
                pdb_text = r3.text
                view = py3Dmol.view(width=700, height=500)
                view.addModel(pdb_text, "pdb")
                view.setStyle({'cartoon': {'color':'spectrum'}})
                view.zoomTo()
                view.show()
                st.components.v1.html(view.js(), height=500)
            else:
                st.warning("PDB structure not found.")

    # ----- Tab 4: Interactive Protein Network -----
    with tabs[3]:
        st.subheader("🎨 Interactive Protein Network")

        # Map protein to x-position
        protein_list = list(proteins)
        pos = {p:i for i,p in enumerate(protein_list)}
        values = [closeness[p] for p in protein_list]
        vmax, vmin = max(values), min(values)

        fig, ax = plt.subplots(figsize=(12,6))

        # Draw edges
        for node in protein_list:
            for neighbor in adj[node]:
                ax.plot([pos[node], pos[neighbor]], [0,0], color='#87CEFA', alpha=0.5, linewidth=1)

        # Draw nodes with color based on closeness
        for i, node in enumerate(protein_list):
            norm_val = (closeness[node]-vmin)/(vmax-vmin+1e-9)
            ax.scatter(pos[node], 0, color=plt.cm.viridis(norm_val), s=150, edgecolor='black', zorder=3)
            ax.text(pos[node], 0.02, node, rotation=90, fontsize=9, ha='center', va='bottom', zorder=4)

        ax.set_title("Interactive Protein Network (Color = Closeness)", fontsize=16, color='#4B0082')
        ax.axis('off')
        st.pyplot(fig)

        # Select a protein for metrics & details
        selected_node = st.selectbox("Select a protein to view metrics", protein_list)
        if selected_node:
            st.markdown(f"### Metrics for **{selected_node}**")
            st.write(f"**Closeness:** {closeness[selected_node]:.4f}")
            st.write(f"**Betweenness:** {betweenness[selected_node]:.4f}")
            st.write(f"**Clustering Coefficient:** {clustering[selected_node]:.4f}")

            col1, col2 = st.columns(2)
            with col1:
                uni_url = f"https://rest.uniprot.org/uniprotkb/{selected_node}.fasta"
                r = requests.get(uni_url)
                if r.ok:
                    seq = "".join(r.text.split("\n")[1:])
                    st.text_area("Protein Sequence (FASTA)", seq, height=200)
                else:
                    st.warning("Sequence not found in UniProt.")

            with col2:
                ps_url = f"https://www.ebi.ac.uk/proteins/api/proteins/{selected_node}"
                headers = {"Accept": "application/json"}
                r2 = requests.get(ps_url, headers=headers)
                if r2.ok:
                    info = r2.json()
                    features = info.get("features", [])
                    motifs = [f"{f['type']}: {f['description']}" for f in features]
                    st.markdown("<h4 style='color:#4B0082;'>Motifs/Domains</h4>", unsafe_allow_html=True)
                    for m in motifs:
                        st.markdown(f"<p style='background-color:#E6E6FA; padding:5px; border-radius:5px;'>{m}</p>", unsafe_allow_html=True)
                else:
                    st.warning("Motifs/Domains not found.")

else:
    st.info("Please upload a CSV file with at least 'Protein1', 'Protein2', and optional 'TaxID' columns.")
