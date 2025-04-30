import streamlit as st
from PIL import Image
import time
import random, os
import pandas as pd

# def inference(agent, image, user_prompt):
def inference(agent, image, user_prompt):
    # best_video, best_prompt, clip_score, tc_score, dd_score
    return agent.infer(image, user_prompt)


# add new annotation record to the existing xx_score.csv
def add_new_record(clip_score, tc_score, dd_score, user_value, df_scores):
    
    if clip_score and tc_score and dd_score and user_value:
        file_name_postfix = time.strftime("%Y%m%d%H%M%S")
        df_scores.loc[df_scores.shape[0]] = ["user_generated" + file_name_postfix, 
                                                                clip_score,
                                                                tc_score,
                                                                dd_score,
                                                                user_value]
        df_scores.to_csv(f"user_marks/user_{st.session_state.current_user}_scores.csv", index=False)

def get_score_df():
    score_file = f"user_marks/user_{st.session_state.current_user}_scores.csv"
    if os.path.exists(score_file):
        df_scores = pd.read_csv(score_file)
    else:
        if not os.path.exists("user_marks"):
            os.makedirs("user_marks")
        df_scores = pd.DataFrame(columns=["file_name", "clip_tva_score", "temporal_consistency", "dynamic_degree", "user_value"])
        df_scores.to_csv(score_file, index=False)
    return df_scores

def on_file_upload():
    st.session_state.user_input = "" 
        

def show(agent):
    st.subheader("🖼️ Upload Image and Add Prompt")
    
    # some session_state data to record the status of the page to avoid data lost when refreshing the page
    if "current_user" not in st.session_state:
        user_id = time.strftime("%Y%m%d%H%M%S")
        st.session_state.current_user = user_id
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
    if "show_score_area" not in st.session_state:
        st.session_state.show_score_area = False
    if "pre_prompt_text" not in st.session_state:
        st.session_state.pre_prompt_text = ""
         
    if "current_video" not in st.session_state:
        st.session_state.current_video = None
    if "best_prompt" not in st.session_state:
        st.session_state.best_prompt = ""
        
    # Image uploader
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], on_change=on_file_upload)

    if uploaded_file:
        st.image(uploaded_file, caption="Image uploaded!", use_container_width=True)
        img = Image.open(uploaded_file).convert("RGB")

        user_prompt = st.text_input("Enter a prompt for the video you'd like to generate: ", key="user_input")
        clip_score = tc_score = dd_score = None
        if user_prompt:
            if st.session_state.pre_prompt_text != user_prompt:
                st.session_state.pre_prompt_text = user_prompt
                msg = st.info("Generating video ...")
                
                best_video, best_prompt, clip_score, tc_score, dd_score = inference(agent, img, user_prompt)
                if best_video != "":
                    st.session_state.current_video = best_video
                    st.session_state.best_prompt = best_prompt
                msg.success("Video generated! ▶️ ")
                
            if st.session_state.current_video != "":
                st.video(open(st.session_state.current_video, 'rb').read())
                st.markdown(f"🚀 LLM refined prompt: {st.session_state.best_prompt}")
    
            if random.randint(0, 2) == 0 or st.session_state.show_score_area:
                st.subheader("✨ We appreciate if you could rate the video quality.")
                st.session_state.show_score_area = True
                slider_value = st.slider("**0 - poorest quality, 10 - highest quality**", min_value=0.0, max_value=10.0, step=0.1, key="user_value")
                
                if st.button("Confirm Score", key="confirm_btn"):
                    if "df_scores" not in st.session_state:
                        st.session_state.df_scores = get_score_df()
                    add_new_record(clip_score, tc_score, dd_score, slider_value, st.session_state.df_scores)

                    msg2 = st.success("Thank you for your rating!")
                    time.sleep(1.5)
                    msg2.empty()  
                    st.session_state.show_score_area = False 

            
                
