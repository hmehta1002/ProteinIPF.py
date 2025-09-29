
import streamlit as st
import pandas as pd
import requests
import networkx as nx
import matplotlib.pyplot as plt
import py3Dmol
import io
import json

st.set_page_config(page_title="Protein Network Explorer", layout="wide")

# ---------------------------- Dark Tab Styling ----------------------------
st.markdown("""
<style>
div[class*="stTabs"] button {
    color: #ffffff;
    background-color: #111111;
}
div[class*="stTabs"] button:hover {
    background-color: #333333;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------- Tabs ----------------------------
tabs = st.tabs(["Upload / Files", "Network Map", "Network Metrics", "Protein Details", 
                "Sequences / FASTA", "Motifs / Domains", "3D Viewer", "Disorder & Stability", 
                "DMNFZ Explorer", "DFWMIN Explorer", "Centrality Measures", "Community Detection", 
                "Network Robustness", "Seq-Structure Mapping", "Downloads / Exports"])

# ---------------------------- Tab 1: Upload ----------------------------
with tabs[0]:
    st.header("Upload your protein/network files")
    uploaded_file = st.file_uploader("Upload CSV or FASTA", type=["csv","fasta"])
    sample_data = pd.DataFrame({"Protein1":["P53","BRCA1"],"Protein2":["MDM2","RAD51"]})
    if st.button("Use Sample Data"):
        df = sample_data.copy()
        st.dataframe(df)
    elif uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
        else:
            seqs = uploaded_file.read().decode().split('>')
            seq_dict = {}
            for s in seqs[1:]:
                lines = s.split('\n')
                header = lines[0].strip()
                sequence = ''.join(lines[1:]).strip()
                seq_dict[header] = sequence
            st.write(seq_dict)

# ---------------------------- Helper Functions ----------------------------
def fetch_uniprot_motifs(protein_id):
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}.json"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        features = data.get('features', [])
        motifs = [f for f in features if f['type'] in ['DOMAIN','MOTIF']]
        return motifs
    else:
        return []

def fetch_alphafold_structure(protein_id):
    # AlphaFold PDB URL
    pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v4.pdb"
    resp = requests.get(pdb_url)
    if resp.status_code == 200:
        return resp.text
    else:
        return None

# ---------------------------- Tab 2: Network Map ----------------------------
with tabs[1]:
    st.header("Network Visualization")
    if 'df' in locals():
        G = nx.from_pandas_edgelist(df, 'Protein1', 'Protein2')
        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(8,6))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1500, font_size=12)
        st.pyplot(plt)

# ---------------------------- Tab 3: Network Metrics ----------------------------
with tabs[2]:
    st.header("Network Metrics")
    if 'G' in locals():
        metrics = pd.DataFrame({
            'Protein': list(G.nodes()),
            'Degree': [G.degree(n) for n in G.nodes()],
            'Betweenness': [nx.betweenness_centrality(G)[n] for n in G.nodes()],
            'Closeness': [nx.closeness_centrality(G)[n] for n in G.nodes()]})
        st.dataframe(metrics)

# ---------------------------- Tab 4: Protein Details ----------------------------
with tabs[3]:
    st.header("Protein Lookup")
    protein_id = st.text_input("Enter UniProt ID for details")
    if protein_id:
        motifs = fetch_uniprot_motifs(protein_id)
        st.write("Motifs/Domains:")
        st.json(motifs)

# ---------------------------- Tab 5: Sequences / FASTA ----------------------------
with tabs[4]:
    st.header("Sequences")
    if uploaded_file and uploaded_file.name.endswith('.fasta'):
        st.write(seq_dict)

# ---------------------------- Tab 6: Motifs / Domains ----------------------------
with tabs[5]:
    st.header("Motifs & Domains")
    if protein_id:
        for f in motifs:
            st.write(f"{f['type']}: {f.get('description','')} ({f.get('begin','')} - {f.get('end','')})")

# ---------------------------- Tab 7: 3D Viewer (AlphaFold) ----------------------------
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

# ---------------------------- Tab 8: Disorder & Stability ----------------------------
with tabs[7]:
    st.header("Intrinsic Disorder & Stability")
    st.write("Currently showing mock disorder for demonstration.")
    import numpy as np
    x = np.arange(0, 100)
    y = np.random.rand(100)
    plt.figure(figsize=(10,4))
    plt.plot(x,y,color='orange')
    plt.title("Disorder Plot")
    st.pyplot(plt)

# ---------------------------- Tab 9 & 10: DMNFZ & DFWMIN ----------------------------
with tabs[8]:
    st.header("DMNFZ Explorer")
    st.write("Novel network feature: DMNFZ algorithm applied.")
    st.write("[Algorithm working placeholder for now]")

with tabs[9]:
    st.header("DFWMIN Explorer")
    st.write("Novel network feature: DFWMIN algorithm applied.")
    st.write("[Algorithm working placeholder for now]")

# ---------------------------- Tab 11: Advanced Centrality ----------------------------
with tabs[10]:
    st.header("Advanced Centrality Measures")
    if 'G' in locals():
        eigen = nx.eigenvector_centrality(G, max_iter=500)
        st.write(pd.DataFrame({'Protein': list(eigen.keys()),'Eigenvector': list(eigen.values())}))

# ---------------------------- Tab 12: Community Detection ----------------------------
with tabs[11]:
    st.header("Community Detection")
    from networkx.algorithms import community
    if 'G' in locals():
        communities = community.label_propagation_communities(G)
        comm_dict = {i+1:list(c) for i,c in enumerate(communities)}
        st.json(comm_dict)

# ---------------------------- Tab 13: Network Robustness ----------------------------
with tabs[12]:
    st.header("Network Robustness Simulation")
    if 'G' in locals():
        G_temp = G.copy()
        degrees = dict(G_temp.degree())
        top_node = max(degrees, key=degrees.get)
        G_temp.remove_node(top_node)
        plt.figure(figsize=(8,6))
        nx.draw(G_temp, pos, with_labels=True, node_color='red', edge_color='gray', node_size=1500)
        st.pyplot(plt)

# ---------------------------- Tab 14: Sequence-Structure Mapping ----------------------------
with tabs[13]:
    st.header("Sequence-Structure Mapping")
    st.write("Mapping sequence disorder onto network positions.")

# ---------------------------- Tab 15: Downloads / Exports ----------------------------
with tabs[14]:
    st.header("Download Metrics & Graphs")
    if 'metrics' in locals():
        csv = metrics.to_csv(index=False).encode()
        st.download_button(label="Download Metrics CSV", data=csv, file_name="metrics.csv", mime='text/csv')
