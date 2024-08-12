import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from feature_extraction_new import extract_features, reshape_features
from model_classification import ClassificationModel
from tqdm import tqdm

class SnoreNoiseDataset(Dataset):
    def __init__(self, dataset_path, sr=16000):
        self.files = []
        self.labels = []
        for label in ['0', '1']:
            class_path = os.path.join(dataset_path, label)
            for file in os.listdir(class_path):
                if file.endswith('.wav'):
                    self.files.append(os.path.join(class_path, file))
                    self.labels.append(int(label))
        self.sr = sr

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        features = extract_features(audio_path, sr=self.sr)
        features = reshape_features(features)  # Ensure correct shape
        features = torch.tensor(features, dtype=torch.float32)
        return features, torch.tensor(label, dtype=torch.long)

def train_model(model, train_loader, val_loader, num_epochs, patience, model_path):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)
    best_val_loss = float('inf')
    early_stop_count = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            early_stop_count = 0
        else:
            early_stop_count += 1
            
        if early_stop_count >= patience:
            print("Early stopping")
            break

if __name__ == "__main__":
    dataset_path = "./Dataset"
    model_path = "./classification_model.pth"
    num_epochs = 20
    patience = 10
    
    train_dataset = SnoreNoiseDataset(os.path.join(dataset_path, "Train/original"))
    val_dataset = SnoreNoiseDataset(os.path.join(dataset_path, "Val/original"))
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationModel().to(device)
    
    train_model(model, train_loader, val_loader, num_epochs, patience, model_path)