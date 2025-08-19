import streamlit as st
import pandas as pd
import requests
import py3Dmol
import matplotlib.pyplot as plt

# -------------------------------
# Utility: Render structure (PDB or AlphaFold) with ligands
# -------------------------------
def render_structure(uniprot_id, pdb_id=None, variants=None):
    st.subheader("Protein Structure Viewer")

    if pdb_id:
        pdb_url = f"https://files.rcsb.org/view/{pdb_id}.pdb"
        response = requests.get(pdb_url)

        if response.ok:
            pdb_data = response.text
            viewer = py3Dmol.view(width=600, height=400)
            viewer.addModel(pdb_data, "pdb")
            viewer.setStyle({"cartoon": {"color": "spectrum"}})

            # Variant highlighting (red sticks)
            if variants:
                for v in variants:
                    try:
                        viewer.addStyle({"resi": int(v)}, {"stick": {"color": "red"}})
                    except:
                        pass

            # Ligand highlighting (yellow spheres)
            for line in pdb_data.splitlines():
                if line.startswith("HETATM"):
                    try:
                        resi = int(line[22:26].strip())
                        viewer.addStyle({"resi": resi}, {"sphere": {"color": "yellow", "radius": 0.5}})
                    except:
                        pass

            viewer.zoomTo()
            st.components.v1.html(viewer._make_html(), height=500)
            st.markdown(f"🔗 [Open in RCSB PDB Viewer](https://www.rcsb.org/structure/{pdb_id})")
        else:
            st.error("Could not fetch PDB file.")
    else:
        # AlphaFold fallback
        st.warning("No experimental structure in PDB. Showing AlphaFold model instead.")
        af_url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        response = requests.get(af_url)

        if response.ok:
            pdb_data = response.text
            viewer = py3Dmol.view(width=600, height=400)
            viewer.addModel(pdb_data, "pdb")
            viewer.setStyle({"cartoon": {"color": "spectrum"}})

            # Variant highlighting
            if variants:
                for v in variants:
                    try:
                        viewer.addStyle({"resi": int(v)}, {"stick": {"color": "red"}})
                    except:
                        pass

            viewer.zoomTo()
            st.components.v1.html(viewer._make_html(), height=500)
            st.markdown(
                f"🔗 [Open in AlphaFold Viewer](https://www.alphafold.ebi.ac.uk/entry/{uniprot_id})"
            )

            # Extract pLDDT scores
            plDDT_scores = []
            for line in pdb_data.splitlines():
                if line.startswith("ATOM"):
                    try:
                        score = float(line[60:66].strip())
                        plDDT_scores.append(score)
                    except:
                        pass

            if plDDT_scores:
                st.subheader("Intrinsic Disorder / Stability Mapping")
                fig, ax = plt.subplots()
                ax.plot(range(1, len(plDDT_scores) + 1), plDDT_scores, color="purple")
                ax.set_xlabel("Residue Index")
                ax.set_ylabel("pLDDT Confidence (0–100)")
                ax.set_title("Predicted Stability / Disorder by Residue")
                st.pyplot(fig)
        else:
            st.error("AlphaFold model not available.")

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.set_page_config(page_title="IPF Protein Explorer", layout="wide")
st.title("🫁 IPF Protein Structure Explorer")
st.markdown("""
This novel app helps researchers explore **Idiopathic Pulmonary Fibrosis (IPF)-related proteins**:  
- 🔍 Query by **protein name or UniProt ID**  
- 📂 Upload CSV with **miRNA–Gene–Protein** interactions  
- 🎨 Visualize **PDB or AlphaFold models**  
- 🧬 Highlight **variants**  
- 🟡 Highlight **ligands**  
- 🌀 Map **intrinsic disorder regions** using AlphaFold pLDDT scores
""")

# -------------------------------
# Single Query Mode
# -------------------------------
st.header("🔍 Single Protein Search")

query = st.text_input("Enter Protein name or UniProt ID (e.g., PTEN, AKT1, Q9HC84)")
variants_input = st.text_input("Enter comma-separated variant residue numbers (optional)")
variants = [v.strip() for v in variants_input.split(",")] if variants_input else None

if st.button("Search Protein"):
    if query:
        query_upper = query.upper()
        pdb_lookup = {"PTEN": "1D5R", "AKT1": "4EKL"}
        pdb_id = pdb_lookup.get(query_upper, None)
        render_structure(query_upper, pdb_id, variants)
    else:
        st.error("Please enter a valid protein name or UniProt ID.")

# -------------------------------
# Batch Mode: CSV Upload
# -------------------------------
st.header("📂 Batch Analysis (CSV Upload)")
uploaded_file = st.file_uploader("Upload CSV with columns: miRNA,Gene,Protein", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)

        for i, row in df.iterrows():
            st.markdown(f"### {row['Protein']} ({row['Gene']}, {row['miRNA']})")
            protein = str(row['Protein']).upper()
            pdb_lookup = {"PTEN": "1D5R", "AKT1": "4EKL"}
            pdb_id = pdb_lookup.get(protein, None)
            render_structure(protein, pdb_id)

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
