import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
from pathlib import Path
import os, sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import json

# Define the simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, 16)   # First hidden layer: 3 -> 16
        self.fc2 = nn.Linear(16, 8)   # Second hidden layer: 16 -> 8
        self.fc3 = nn.Linear(8, 1)    # Output layer: 8 -> 1

    def forward(self, x):
        x = torch.relu(self.fc1(x))   # Activation after 1st layer
        x = torch.relu(self.fc2(x))   # Activation after 2nd layer
        x = self.fc3(x)               # No activation at output
        x = torch.clamp(x, 0, 10)
        # x = torch.sigmoid(self.fc3(x)) * 10  # Sigmoid output scaled to [0, 10]
        return x

def train_NN(train_df, test_df, user_stats, model_saved_path):
            
    X_train = train_df[['clip_tva_score', 'temporal_consistency', 'dynamic_degree']]
    # y_truth = reward_data[['final_score']]
    y_train = train_df[['final_score']]
    
    X_test = test_df[['clip_tva_score', 'temporal_consistency', 'dynamic_degree']]
    y_test = test_df[['final_score']]
        
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled , dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
    model = SimpleNN()
    # Loss and optimizer
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    losses = []
    
    # Training loop
    epochs = 200
    for epoch in range(epochs):
        model.train()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # <-- save loss

        # Print every 50 epochs
        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
    print("finish training ...")
    model_saved_path = model_saved_path + '/reward_model_state_dict_NN.pth'
    torch.save(model.state_dict(), model_saved_path)
        
    # return model
    model.eval()
    prediction = model(X_test_tensor).detach().numpy() 
        
    test_result = X_test.copy()
    test_result['y_real'] = y_test_tensor
    
    # y_pred_transformed = []
    # # for index, pred in enumerate(prediction.detach().numpy()):
    # for index, pred in enumerate(prediction):
    #     user_id = test_df.iloc[index]["user_id"]
    #     user_min = user_stats[user_id][0]
    #     user_std_or_range = user_stats[user_id][1]
    #     y_pred_transformed.append(user_min + pred * user_std_or_range)
        
    # test_result['y_pred'] = y_pred_transformed 
    test_result['y_pred'] = prediction
    
    # Plot loss curve
    plt.plot(losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
    
    # return model_saved_path, test_result, mean_squared_error(y_test, y_pred_transformed)
    return model_saved_path, test_result, mean_squared_error(y_test, prediction)


def train_LR(train_df, test_df, user_stats, save_path):
    
    
    X_train = train_df[['clip_tva_score', 'temporal_consistency', 'dynamic_degree']]
    y_train = train_df[['final_score']]
    
    X_test = test_df[['clip_tva_score', 'temporal_consistency', 'dynamic_degree']]
    y_test = test_df[['final_score']]
        
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    prediction = model.predict(X_test_scaled)
    
    model_path = save_path + '/reward_model_LR.pkl'
    # save
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)
        
    test_result = X_test.copy()
    test_result['y_real'] = y_test
    
    y_pred_transformed = []
    for index, pred in enumerate(prediction):
        user_id = test_df.iloc[index]["user_id"]
        user_min = user_stats[user_id][0]
        user_std_or_range = user_stats[user_id][1]
        y_pred_transformed.append(user_min + pred * user_std_or_range)
        
    test_result['y_pred'] = y_pred_transformed 
    print(test_result)
    
    return model_path, test_result, mean_squared_error(y_test, prediction)
    

def draw_chart(test_df):
    fig, ax = plt.subplots()
    
    ax.scatter(range(1, test_df.shape[0]+1), test_df['y_real'], color='blue', alpha=0.7, label="Y Real")
    ax.scatter(range(1, test_df.shape[0]+1), test_df['y_pred'], color='red', alpha=0.7, label="Y Pred")
    ax.set_title('True Value & Prediction')
    ax.set_xlabel('True Value')
    ax.set_ylabel('Prediction')
    plt.legend()
    plt.show()

    
    
def train_workflow():
    
    cur_dir = Path(sys.argv[0])
    base_dir = str(cur_dir.parent.parent.resolve())
    model_path = base_dir + "/user_marks/model"
    
    if os.path.exists(model_path + "/reward_df.csv") and os.path.exists(model_path + "/reward_test_df.csv"):
        train_df = pd.read_csv(model_path + "/reward_df.csv")
        test_df = pd.read_csv(model_path + "/reward_test_df.csv")
        
        with open(model_path + "/user_stats.json", "r") as f:
            user_stats = json.load(f)
        
        # choose either NN or linear regression
        model_path, test_result, mse = train_NN(train_df, test_df, user_stats, model_path)
        # model_path, test_result, mse = train_LR(train_df, test_df, user_stats, model_path)
        
        # draw_chart(test_result)
        return model_path, test_result, mse 
    else:
        print("directory does not exist.")
        
if __name__ == "__main__":
    train_workflow()
        

        
