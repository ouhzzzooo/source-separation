import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from feature_extraction_new import extract_features, reshape_features
from model_classification import ClassificationModel

class SnoreNoiseTestDataset(Dataset):
    def __init__(self, dataset_path, feature_set, label=1, sr=16000):
        self.files = []
        self.labels = []
        if os.path.isdir(dataset_path):
            for file in os.listdir(dataset_path):
                if file.endswith('.wav'):
                    self.files.append(os.path.join(dataset_path, file))
                    self.labels.append(label)
        else:
            raise FileNotFoundError(f"No such directory: {dataset_path}")
        self.sr = sr
        self.feature_set = feature_set

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        label = self.labels[idx]
        features = extract_features(audio_path, sr=self.sr)
        features = reshape_features(features, feature_set=self.feature_set)
        features = torch.tensor(features, dtype=torch.float32)
        return features, torch.tensor(label, dtype=torch.long)

def evaluate_model(model, test_loader):
    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    return test_preds, test_targets

if __name__ == "__main__":
    test_data_path = "./Dataset/Test/original"
    reconstructed_data_path = "./Reconstructed/reconstructed_data"
    model_path = "./classification_model.pth"
    feature_sets = ['mfccs', 'spectral', 'rmse', 'zcr', 'combined']
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationModel()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    
    results = []
    for feature_set in feature_sets:
        # Load datasets
        test_dataset = SnoreNoiseTestDataset(test_data_path, feature_set=feature_set)
        reconstructed_dataset = SnoreNoiseTestDataset(reconstructed_data_path, feature_set=feature_set)

        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        reconstructed_loader = DataLoader(reconstructed_dataset, batch_size=16, shuffle=False)

        # Evaluate model
        test_preds, test_targets = evaluate_model(model, test_loader)
        reconstructed_preds, reconstructed_targets = evaluate_model(model, reconstructed_loader)

        # Compute metrics for test data
        accuracy = accuracy_score(test_targets, test_preds)
        precision = precision_score(test_targets, test_preds, average='binary')
        recall = recall_score(test_targets, test_preds, average='binary')
        f1 = f1_score(test_targets, test_preds, average='binary')
        cm = confusion_matrix(test_targets, test_preds)

        # Append results
        results.append({
            'Feature Set': feature_set,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Confusion Matrix': cm
        })

    # Create a DataFrame to display the results
    results_df = pd.DataFrame(results)
    print(results_df)