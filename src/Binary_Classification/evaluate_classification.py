import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from feature_extraction_new import extract_features
from model_classification import ClassificationModel

class SnoreNoiseTestDataset(Dataset):
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
        features = torch.tensor(features, dtype=torch.float32)
        return features, torch.tensor(label, dtype=torch.long)

def evaluate_model(model, test_loader):
    model.eval()
    test_preds, test_targets = [], []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(test_targets, test_preds)
    precision = precision_score(test_targets, test_preds, average='binary')
    recall = recall_score(test_targets, test_preds, average='binary')
    f1 = f1_score(test_targets, test_preds, average='binary')
    cm = confusion_matrix(test_targets, test_preds)
    
    return accuracy, precision, recall, f1, cm

if __name__ == "__main__":
    test_data_path = "./Dataset/Test/original"
    reconstructed_data_path = "./Reconstructed/reconstructed_data"
    model_path = "./classification_model.pth"
    
    # Load datasets
    test_dataset = SnoreNoiseTestDataset(test_data_path)
    reconstructed_dataset = SnoreNoiseTestDataset(reconstructed_data_path)

    if len(test_dataset) == 0 or len(reconstructed_dataset) == 0:
        raise ValueError("No data found. Please check the dataset paths and ensure there are .wav files in the directories.")

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    reconstructed_loader = DataLoader(reconstructed_dataset, batch_size=16, shuffle=False)
    
    # Load model
    model = ClassificationModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    # Evaluate model
    test_metrics = evaluate_model(model, test_loader)
    reconstructed_metrics = evaluate_model(model, reconstructed_loader)
    
    # Create a table to compare the results
    metrics_table = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Original': [test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3]],
        'Reconstructed': [reconstructed_metrics[0], reconstructed_metrics[1], reconstructed_metrics[2], reconstructed_metrics[3]]
    })

    print("Confusion Matrix for Original Test Data:")
    print(test_metrics[4])
    print("\nConfusion Matrix for Reconstructed Test Data:")
    print(reconstructed_metrics[4])
    
    print("\nEvaluation Metrics:")
    print(metrics_table)