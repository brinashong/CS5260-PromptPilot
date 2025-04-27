import streamlit as st
import os
import pandas as pd
import reward_model

                
def get_score_files(SCORE_DIR):

    users = {}
    for file in os.listdir(SCORE_DIR):
        if file[-4:] == ".csv" and file[:4] == "user":
            item_path = os.path.join(SCORE_DIR, file)
            user_name = file.split("_")[1]
            
            # put one user's score in one dictionary item
            df = pd.read_csv(item_path)
            if user_name not in users:
                users[user_name] = [df]
            else:
                users[user_name].append(df)
    return users


def normalize_user_scores(scores_df):
    normalized_scores = []
    normalized_user_stats = {}
    for user_id, group in scores_df.groupby('user_id'):
        if len(group) >= 500:
            # Z-score
            mean = group['score'].mean()
            std = group['score'].std()
            normalized_user_stats[user_id] = (mean, std)
            group['norm_score'] = (group['score'] - mean) / std
        else:
            # Min-max
            min_score = group['score'].min()
            max_score = group['score'].max()
            group['norm_score'] = (group['score'] - min_score) / (max_score - min_score + 1e-8) * 10
        normalized_scores.append(group)
    return pd.concat(normalized_scores, axis=0, ignore_index=True), normalized_user_stats


def preprocessing(users:dict):
    all_user_scores = []
    for user, user_scores in dict(users).items():        
        if len(user_scores) > 1:
            user_scores_df = pd.concat(user_scores, axis=0, ignore_index=True)
        else:
            user_scores_df = user_scores[0]
        
        # get average score for repeated image for single user and then add user_id series
        averaged_user_scores = user_scores_df.groupby(['file_name', 'clip_tva_score', 'temporal_consistency', 'dynamic_degree']).agg(avg_score=('user_value', 'mean')).reset_index().rename(columns={"avg_score": "score"})
        averaged_user_scores['user_id'] = user
        all_user_scores.append(averaged_user_scores)
        all_scores = pd.concat(all_user_scores, axis=0, ignore_index=True)

    
    # get normalized score for a single user and then concatenate all users' normalized scores into one dataframe
    normalized_scores_df, normalized_user_stats = normalize_user_scores(all_scores)
    final_image_scores_df = (
    normalized_scores_df
    .groupby(['file_name', 'clip_tva_score', 'temporal_consistency', 'dynamic_degree'])['norm_score']
    .median()  # or .mean(), depending on your needs
    .reset_index()
    .rename(columns={"norm_score": "final_score"})
    )
    
    return final_image_scores_df, normalized_user_stats

def cleanup_callback():
    # if "processed_result" in st.session_state:
    #     st.session_state.processed_result.empty()
    
    if "test_result" in st.session_state:
        st.session_state.test_result.empty()
    
    
    
def show():
    USER_SCORE_DIR = st.text_input("Input the user score directory: ", key="user_score_dir", on_change=cleanup_callback())
    
    
    if USER_SCORE_DIR != "" and os.path.isdir(USER_SCORE_DIR):
        users = get_score_files(USER_SCORE_DIR) 
        print("refresh!")
        
        if len(users) == 0:
            st.warning("There is no user score data in this directory. Please choose again.")
        else:
            data_df, user_stats = preprocessing(users)
            
            saved_path = USER_SCORE_DIR + "/reward_df.csv"
            data_df.to_csv(saved_path, index=False)
            st.dataframe(data_df, key="processed_result")
            st.success(f"The preprocessed data has been saved to {saved_path}. Do you want to train the model?")
            
            if st.button("Yes, please train the model now!"):
                msg = st.info("training is in progress...")
                model_path, test_df = reward_model.train_workflow()
                msg.info(f"training is done! The model is saved to {model_path}")
                
                st.dataframe(test_df, key="test_result")
            
    
            