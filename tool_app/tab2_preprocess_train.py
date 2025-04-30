import streamlit as st
import os
import pandas as pd
# import tool_app.reward_model as reward_model
import reward_model
import matplotlib.pyplot as plt
import random, json
                
def get_score_files(SCORE_DIR):

    users = {}
    users_test = {}
    for file in os.listdir(SCORE_DIR):
        if file[-4:] == ".csv" and file[:4] == "user":
            item_path = os.path.join(SCORE_DIR, file)
            user_name = file.split("_")[1]
            
            # put one user's score in one dictionary item
            df = pd.read_csv(item_path)
            df['user_id'] = user_name
            
            # handle some very small dataset
            if df.shape[0] <= 2:
                continue
            elif df.shape[0] <= 5:
                test_indices = random.sample(range(0, df.shape[0]), 1)
            else:    
                test_indices = random.sample(range(0, df.shape[0]), int(len(df) * 0.2))
            
            df_test = df.iloc[test_indices]
            df_train = df.drop(test_indices)
            
            if user_name not in users:
                users[user_name] = [df_train]
                users_test[user_name] = [df_test]
            else:
                users[user_name].append(df_train)
                users_test[user_name].append(df_test)
    # separate training and test datasets to avoid data leakage in preprocessing stage
    return users, users_test


def normalize_user_scores(scores_df):
    normalized_scores = []
    normalized_user_stats = {}
    for user_id, group in scores_df.groupby('user_id'):
        if len(group) >= 30:
            # Z-score
            mean = group['score'].mean()
            std = group['score'].std()
            normalized_user_stats[user_id] = (mean, std)
            group['norm_score'] = (group['score'] - mean) / std
        else:
            # Min-max
            min_score = group['score'].min()
            max_score = group['score'].max()
            group['norm_score'] = (group['score'] - min_score) / (max_score - min_score + 1e-8)
            normalized_user_stats[user_id] = (min_score, max_score - min_score + 1e-8)
        normalized_scores.append(group)
    return pd.concat(normalized_scores, axis=0, ignore_index=True), normalized_user_stats


def preprocessing(users:dict):
    all_user_scores = []
    for user, user_scores in dict(users).items():        
        if len(user_scores) > 1:
            user_scores_df = pd.concat(user_scores, axis=0, ignore_index=True)
        else:
            user_scores_df = user_scores[0]
        
        # get average score for one single image (since one user might score the same file multiple times) for single user and then add user_id series
        averaged_user_scores = user_scores_df.groupby(['file_name', 'clip_tva_score', 'temporal_consistency', 'dynamic_degree']).agg(avg_score=('user_value', 'mean')).reset_index().rename(columns={"avg_score": "score"})
        averaged_user_scores['user_id'] = user
        all_user_scores.append(averaged_user_scores)
        all_scores = pd.concat(all_user_scores, axis=0, ignore_index=True)

    # get normalized score for a single user and then concatenate all users' normalized scores into one dataframe
    normalized_scores_df, normalized_user_stats = normalize_user_scores(all_scores)
    final_image_scores_df = (
    normalized_scores_df
    .groupby(['file_name', 'clip_tva_score', 'temporal_consistency', 'dynamic_degree'])['norm_score']
    .mean() 
    .reset_index()
    .rename(columns={"norm_score": "final_score"})
    )
    return final_image_scores_df, normalized_user_stats


def preprocessing_test_df(users_test:dict):
    all_user_scores = []
    for user, user_scores in dict(users_test).items():
        if len(user_scores) > 1:
            user_scores_df = pd.concat(user_scores, axis=0, ignore_index=True)
        else:
            user_scores_df = user_scores[0]
        user_scores_df['user_id'] = user
        user_scores_df = user_scores_df.rename(columns={'user_value': 'final_score'})
        
        all_user_scores.append(user_scores_df)
    return pd.concat(all_user_scores, axis=0, ignore_index=True)
    

def cleanup_callback():
    if "test_result" in st.session_state:
        st.session_state.test_result.empty()
    

def draw_chart(test_df):
    fig, ax = plt.subplots(figsize=(5, 3))
    
    ax.scatter(range(1, test_df.shape[0]+1), test_df['y_real'], color='blue', alpha=0.7, label="Y Real", s=5)
    ax.scatter(range(1, test_df.shape[0]+1), test_df['y_pred'], color='red', alpha=0.7, label="Y Pred", s=5)
    ax.set_title('True Value & Prediction')
    ax.set_xlabel('Data Point')
    ax.set_ylabel('User Score')
    plt.legend()
    st.pyplot(fig, clear_figure=True, use_container_width=True)
    
    
def show():
    USER_SCORE_DIR = st.text_input("Input user score directory: ", key="user_score_dir", on_change=cleanup_callback())
    
    if USER_SCORE_DIR != "" and os.path.isdir(USER_SCORE_DIR):
        users, users_test = get_score_files(USER_SCORE_DIR) 
        
        if len(users) == 0:
            st.warning("There is no user score data in this directory. Please choose again.")
        else:
            data_df, user_stats = preprocessing(users)
            data_test_df = preprocessing_test_df(users_test)
            model_path = USER_SCORE_DIR + "/model"
            
            if not os.path.exists(model_path):
                os.mkdir(model_path)
                
            saved_path = model_path + "/reward_df.csv"
            data_df.to_csv(saved_path, index=False)
            
            test_saved_path = model_path + "/reward_test_df.csv"
            data_test_df.to_csv(test_saved_path, index=False)
            
            
            with open(model_path + "/user_stats.json", 'w') as file:
                json.dump(user_stats, file, indent=4)
            
            st.markdown("")
            st.subheader("🎯 Pre-processed Training Dataset (with normalized final_score):")
            st.dataframe(data_df, key="processed_result")
            st.success(f"The preprocessed data has been saved to {saved_path}. \n\nWould you like to train the model now?")
            
            if st.button("Yes, please!"):
                msg = st.info("training is in progress...")
                model_path, test_df, mse = reward_model.train_workflow()
                msg.info(f"Training complete! The model is saved to {model_path}")
                msg.info(f"Model Test MSE: {mse}")
                
                test_df = test_df.sort_values(by='y_real')
                st.dataframe(test_df, key="test_result")
                
                draw_chart(test_df)
                
                
            
    
            