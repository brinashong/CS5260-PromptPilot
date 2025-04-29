import streamlit as st
import json
import cv2
import os, sys, glob
import pandas as pd
import time, random

# /Users/evansun/Documents/Claudia/CS5260_Neural_network_and_deep_learning/CS5260-PromptPilot/PromptPilot Dataset/exp1

JUDGE_SAMPLES = 50

# find all the cascading sob folders under VIDEO_DIR
def get_subfolders(folder_path):
  subfolders = []
  for item in os.listdir(folder_path):
    item_path = os.path.join(folder_path, item)
    if os.path.isdir(item_path):
      subfolders.append(item_path)
      subfolders.extend(get_subfolders(item_path)) # Recursive call for nested subfolders
  return subfolders
            
# add new annotation record to the existing xx_score.csv
def add_new_record(file_name, video_data, user_value, df_scores):
    
    if video_data and user_value:
        df_scores.loc[df_scores.shape[0]] = [file_name, video_data['scores']['clip_tva_score'],
                                                                video_data['scores']['temporal_consistency'],
                                                                video_data['scores']['dynamic_degree'],
                                                                user_value]
        df_scores.to_csv(f"user_marks/user_{st.session_state.judge_name}_scores.csv", index=False)
        st.session_state.confirmed_score = True
    
    
# only filter out those folders which contain .mp4 and scores.json and sample some files     
def get_video_files(VIDEO_DIR):
    folder_names = get_subfolders(VIDEO_DIR)
    video_files = []
    for folder in folder_names:
        matching_files = glob.glob(folder + "/*.mp4")
        if len(matching_files) > 0:
            sorted_files = sorted(matching_files, key=os.path.getctime, reverse=False)
            json_path = folder + "/scores.json"
            with open(json_path, 'r') as f:
                video_data = json.load(f)
            
            if len(matching_files) == 1:
                video_files.append({"video_file": matching_files[0], 
                                    "meta_data": video_data})
            else:
                for i, file in enumerate(sorted_files):
                    video_files.append({"video_file": file, 
                                        "meta_data": video_data[i]})
                
    sample_nums = min(len(video_files), JUDGE_SAMPLES)
    random.seed(time.time())
    judge_videos = random.sample(video_files, sample_nums)
    return judge_videos


# read user_xxxx_scores.csv into dataframe
def get_score_df():
    score_file = f"user_marks/user_{st.session_state.judge_name}_scores.csv"
    if os.path.exists(score_file):
        df_scores = pd.read_csv(score_file)
    else:
        if not os.path.exists("user_marks"):
            os.makedirs("user_marks")
        df_scores = pd.DataFrame(columns=["file_name", "clip_tva_score", "temporal_consistency", "dynamic_degree", "user_value"])
        df_scores.to_csv(score_file, index=False)
    return df_scores


def show():     
    VIDEO_DIR = st.text_input("Input the video directory:", key="video_dir")
    JUDGE_NAME = st.text_input("Input judge name:", key="judge_name")
    
    if VIDEO_DIR != "" and os.path.isdir(VIDEO_DIR) and JUDGE_NAME != "":
        # initialization
        if "video_files" not in st.session_state:
            st.session_state.video_files = get_video_files(VIDEO_DIR)
        if "df_scores" not in st.session_state:
            st.session_state.df_scores = get_score_df()
        if "video_index" not in st.session_state:
            st.session_state.video_index = 0

        st.header("📹 Video Quality Annotation & Training")
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            if st.button("Prev") and st.session_state.video_index > 0:
                st.session_state.video_index -= 1
        with col3:
            if st.button("Next") and st.session_state.video_index < len(st.session_state.video_files) - 1:
                st.session_state.video_index += 1
        with col2:
            current_video = st.session_state.video_files[st.session_state.video_index]
            st.video(open(current_video['video_file'], 'rb').read())
            st.caption(f"Showing video {st.session_state.video_index + 1} of {len(st.session_state.video_files)}")

        # show video meta data (original prompt and metrics)
        if current_video['meta_data'] is not None:
            st.markdown(f"**🙎🏻 User prompt:**")
            st.markdown(f"""
            <div style="padding: 0px; border-radius: 5px; background-color: white; color: blue;">
            {current_video['meta_data']['prompt']}</div>""", unsafe_allow_html=True)
            
            left_half, right_half = st.columns([1, 1])

            with left_half:
                st.subheader("📝 Metrics Info")
                st.markdown(f"**clip_tva_score:**  {str(current_video['meta_data']['scores']['clip_tva_score'])}")
                st.markdown(f"**temporal_consistency:**  {str(current_video['meta_data']['scores']['temporal_consistency'])}")
                st.markdown(f"**dynamic_degree:**:  {str(current_video['meta_data']['scores']['dynamic_degree'])}")
        
            with right_half:
                # show slider bar for user to choose a value
                st.subheader("🎚️ Please mark your score for the video quality")
                slider_value = st.slider("Drag to select a value:", min_value=0.0, max_value=10.0, step=0.1)
            
                # append the score to dataframe
                if st.button("Confirm Your Score"):
                    file_name = current_video['video_file'].rsplit("/", 1)[1][:-4]
                    add_new_record(file_name, current_video['meta_data'], slider_value, st.session_state.df_scores)
                    # if st.session_state.video_index < len(st.session_state.video_files) - 1:
                    #     st.session_state.video_index += 1
                    msg = st.success("Saved the record!")
                    time.sleep(1.5)
                    msg.empty()   
        else:
            st.warning("No meta data for this video.") 
        
