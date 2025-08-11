import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import requests
import tempfile

st.set_page_config(page_title="IPF Protein Explorer", layout="wide")

# ----------------------------
# Functions
# ----------------------------

def load_data(file):
    """Load CSV or JSON protein data."""
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".json"):
        return pd.read_json(file)
    else:
        st.error("❌ Only CSV or JSON files are supported.")
        return None

def build_network(df):
    """Create a basic protein interaction network."""
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['Protein'])
        if 'InteractsWith' in df.columns and pd.notna(row['InteractsWith']):
            for target in str(row['InteractsWith']).split(";"):
                G.add_edge(row['Protein'], target.strip())
    return G

def plot_network(G):
    """Plotly network visualization."""
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none', mode='lines'
    )
    node_x, node_y, node_text = [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text,
        textposition="top center",
        hoverinfo='text', marker=dict(
            showscale=False, color='#1f77b4', size=12, line_width=2
        )
    )
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=0, l=0, r=0, t=0)))
    return fig

def fetch_structure(protein_id):
    """Check PDB first, else AlphaFold."""
    pdb_url = f"https://files.rcsb.org/download/{protein_id}.pdb"
    res = requests.get(pdb_url)
    if res.status_code == 200:
        return pdb_url

    af_url = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v4.pdb"
    res_af = requests.get(af_url)
    if res_af.status_code == 200:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
        tmp.write(res_af.text.encode())
        tmp.flush()
        return af_url
    return None

def show_structure_ngl(pdb_url):
    """Embed NGL.js 3D viewer in Streamlit."""
    ngl_script = f"""
    <div id="viewport" style="width:100%; height:500px;"></div>
    <script src="https://unpkg.com/ngl@latest/dist/ngl.js"></script>
    <script>
      var stage = new NGL.Stage("viewport");
      stage.loadFile("{pdb_url}", {{ ext: "pdb" }}).then(function (o) {{
        o.addRepresentation("cartoon", {{color: "spectrum"}});
        stage.autoView();
      }});
    </script>
    """
    st.components.v1.html(ngl_script, height=520)

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🧬 IPF Protein Explorer")
st.write("Upload protein data, explore networks, and view structures via PDB or AlphaFold.")

uploaded = st.file_uploader("Upload CSV/JSON file with 'Protein' column", type=["csv", "json"])

if uploaded:
    df = load_data(uploaded)
    if df is not None:
        st.subheader("Protein Interaction Network")
        G = build_network(df)
        fig = plot_network(G)
        st.plotly_chart(fig, use_container_width=True)

        protein_choice = st.selectbox("Select a protein to view structure", df['Protein'].unique())
        if st.button("Load Structure"):
            pdb_link = fetch_structure(protein_choice)
            if pdb_link:
                st.success(f"Showing structure for {protein_choice}")
                show_structure_ngl(pdb_link)
            else:
                st.error(f"No structure found for {protein_choice} in PDB or AlphaFold.")
