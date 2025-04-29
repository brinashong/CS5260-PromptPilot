import streamlit as st
import sys
# import tab1_annotation, tab2_preprocess_train as tab2_preprocess_train, tab3_inference
import tab1_annotation, tab2_preprocess_train, tab3_inference
from pathlib import Path
from agent import PromptPilot

import argparse
from omegaconf import OmegaConf
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)
from video_generation import Seine

# cur_dir = Path(sys.argv[0])
# base_dir = str(cur_dir.parent.parent.resolve())
# VIDEO_DIR = base_dir + "/results"
st.set_page_config(page_title="Prompt Pilot", layout="centered")
st.title("Prompt Pilot 🚀")
st.subheader("Steering Your Prompts to High-Fidelity Videos")
# tab1, tab2, tab3 = st.tabs(["📹 Reward Network Annotation", "Reward Annotation Training", "🤖 Video Agent 🤖 "])
tab3, tab1, tab2 = st.tabs(["🎥 Video Generation", "🧑‍⚖️ Human Evaluation", "🧠 Reward Model Training"])
  
if "agent" not in st.session_state:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/seine.yaml")
    args, unknown = parser.parse_known_args()
    omega_conf = OmegaConf.load(args.config)
    seine_model = Seine(omega_conf)
    st.session_state.agent = PromptPilot(seine_model)

with tab1:
    tab1_annotation.show()

with tab2:
    tab2_preprocess_train.show()
     
with tab3:
    tab3_inference.show(st.session_state.agent)