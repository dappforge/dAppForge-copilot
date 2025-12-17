import streamlit as st
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, project_root)

from knowledge_graph_core.kg_rag.inference import claude_inference, composable_graph_inference
from utils.config import configure_settings

# Configure settings
Settings = configure_settings()

def main():
    st.title("dApp Code Generation")
    st.write("Generate code using different blockchain knowledge graphs")

    # Input section
    prefix_code = st.text_area("Enter your code prefix or description")
    kg_name = st.radio("Select Knowledge Graph", ["substrate", "ink", "solidity", "rust"])

    if st.button("Generate Code"):
        if prefix_code:
            try:
                response, edges, subplot = claude_inference(prefix_code, kg_name)
                st.code(response, language="python")
                if subplot:
                    st.write("Knowledge Graph Context:")
                    st.components.v1.html(subplot, height=600)
            except Exception as e:
                st.error(f"Error generating code: {str(e)}")
        else:
            st.warning("Please enter a code prefix or description")

if __name__ == "__main__":
    main()
