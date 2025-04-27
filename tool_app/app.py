import streamlit as st
import sys
import tab1_annotation, tool_app.tab2_preprocess_train as tab2_preprocess_train, tab3_inference
from pathlib import Path


# cur_dir = Path(sys.argv[0])
# base_dir = str(cur_dir.parent.parent.resolve())
# VIDEO_DIR = base_dir + "/results"
st.set_page_config(page_title="Video Generation Tool", layout="centered")
st.title("📂 Multi-tab Streamlit App")
tab1, tab2, tab3 = st.tabs(["📹 Reward Network Annotation", "Reward Annotation Training", "🤖 Video Agent 🤖 "])
  
  
with tab1:
    tab1_annotation.show()

with tab2:
    tab2_preprocess_train.show()
     
with tab3:
    tab3_inference.show()