import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Parameters ---
LEARNING_RATE = 0.0001
VERBOSE = 1000

# Define the neural network
class NeuralNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.hidden1 = nn.Linear(64, 64)
        self.hidden  = nn.Linear(64, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.hidden1(x))
        # x = torch.relu(self.hidden(x))
        x = torch.relu(self.hidden(x))
        x = torch.relu(self.hidden2(x))
        x = self.fc2(x)
        return x

def train_nn(model: nn.Module, X: np.ndarray, y: np.ndarray, num_epochs: int = 1000) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.75)    
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

    model.train()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (epoch+1) % VERBOSE == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            
    return model

# --- Model predition --- #
def model_predict(Cb, w1, w2, model):
    if hasattr(model, 'predict'):
        return model.predict([[w1, Cb]])[0]
    elif isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            return model(torch.tensor([w1, Cb], dtype=torch.float32).to(device)).item()
    else:
        raise ValueError('Model is not supported')