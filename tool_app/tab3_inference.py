import streamlit as st
import threading
from PIL import Image
import time, sys
from pathlib import Path

def inference(agent, image, user_prompt):
    # mock data
    # time.sleep(5)
    # print("inf")
    # return "/Users/evansun/Documents/Claudia/CS5260_Neural_network_and_deep_learning/CS5260-PromptPilot/PromptPilot Dataset/exp1/Outdoor/ancient_wall_besides_town/ancient_wall_besides_town_20250426_022159.mp4"
    
    # best_video, best_prompt, clip_score, tc_score, dd_score
    return agent.infer(image, user_prompt)


cur_dir = Path(sys.argv[0])
base_dir = str(cur_dir.parent.resolve())
VIDEO_DIR = base_dir + "/results"

# Function to reset text when a new image is uploaded
def on_file_upload():
    st.session_state.user_input = ""  # Clear text input


def show(agent):
    st.subheader("🖼️ Upload Image and Add Prompt")
    
    # Image uploader
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], on_change=on_file_upload)
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    if uploaded_file:
        st.session_state.user_input
        st.image(uploaded_file, caption="Image uploaded!", use_container_width=True)
        img = Image.open(uploaded_file).convert("RGB")

        # Text input
        user_prompt = st.text_input("Enter a prompt for the video you'd like to generate: ", key="user_input")
        
        if user_prompt:
            msg = st.info("Generating video ...")
            video_path = inference(agent, img, user_prompt)
            msg.success("Video generated! ▶️ ")
            
            if video_path != "":
                st.video(open(video_path, 'rb').read())
