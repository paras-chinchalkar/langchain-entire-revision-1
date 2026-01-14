import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


torch.manual_seed(42)
np.random.seed(42)

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Split the training and testing here
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    return X_train, X_test, y_train, y_test

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(optimizer_name, X_train, y_train, X_test, y_test, epochs=100, lr=0.01):
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer")
        
    # Train
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = correct / y_test.size(0)
        
    return accuracy

def main():
    try:
        X_train, X_test, y_train, y_test = load_data()
        
        optimizers = ['SGD', 'Adam', 'RMSprop']
        
        print(f"\nComparing Optimizers on Iris Dataset (Epochs=100, LR=0.01)")
        print("-" * 45)
        print(f"{'Optimizer':<15} | {'Accuracy':<10}")
        print("-" * 45)
        
        for opt in optimizers:
            acc = train_model(opt, X_train, y_train, X_test, y_test)
            print(f"{opt:<15} | {acc:.4f}")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
