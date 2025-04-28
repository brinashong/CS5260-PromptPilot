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
import matplotlib.pyplot as plt


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
        return x

def train_NN(X, y, save_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    losses = []  # <-- track loss

    # Training loop
    epochs = 3000
    for epoch in range(epochs):
        model.train()
        #print(epoch)
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print every 50 epochs
        if (epoch+1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
        #print("finish training ...")
        model_path = save_path + '/reward_model_state_dict_NN.pth'
        torch.save(model.state_dict(), model_path)
        
        # return model
        model.eval()
        prediction = model(X_test_tensor)
        losses.append(loss.item())  # <-- save loss

        test_result = X_test.copy()
        test_result['y_real'] = y_test_tensor
        test_result['y_pred'] = prediction.detach().numpy() 
        ##print(test_result)  
        # Plot loss curve
    print(test_result)
    plt.plot(losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()

    plt.figure(figsize=(6,6))
    plt.scatter(test_result['y_real'], test_result['y_pred'], alpha=0.7)
    plt.plot([test_result['y_real'].min(), test_result['y_real'].max()],
         [test_result['y_real'].min(), test_result['y_real'].max()],
         color='red', linestyle='--')  # perfect prediction line  
    plt.xlabel('Ground Truth (y_real)')
    plt.ylabel('Predicted (y_pred)')
    plt.title('Real vs. Predicted Scatter Plot')
    plt.grid(True)
    plt.show()


    return model_path, test_result


def train_LR(X, y, save_path):
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    model_path = save_path + '/reward_model_LR.pkl'
    # save
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)
        
    test_result = X_test.copy()
    test_result['y_real'] = y_test
    test_result['y_pred'] = y_pred
    print(test_result)
    return model_path, test_result

    
    

def train_workflow():
    
    cur_dir = Path(sys.argv[0])
    base_dir = str(cur_dir.parent.parent.resolve())
    score_dir = "/content/user_marks"
    
    if os.path.exists(score_dir ):
        reward_data = pd.read_csv("/content/user_marks/user_marks/user_aa_aascores.csv" )
        
        X= reward_data[['clip_tva_score', 'temporal_consistency', 'dynamic_degree']]
        y = reward_data[['user_value']]
        
        model_path, test_result = train_NN(X, y, score_dir)
        ##model_path, test_result = train_LR(X, y, score_dir)
        return model_path, test_result
    else:
        print("directory does not exist.")
        
if __name__ == "__main__":
    train_workflow()
        

        
