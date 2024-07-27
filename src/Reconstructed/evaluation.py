import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from torch.utils.data import DataLoader, Dataset
from model_reconstructed import get_model

class SnoreDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.size(0), 1, -1)  # Ensure inputs have shape (batch_size, 1, sequence_length)
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Ensure all elements are sequences
    all_preds = [np.atleast_1d(pred) for pred in all_preds]
    all_labels = [np.atleast_1d(label) for label in all_labels]

    # Find the minimum length of the predictions and true labels
    min_length = min(min(map(len, all_preds)), min(map(len, all_labels)))

    # Trim all predictions and labels to the minimum length
    all_preds = np.array([pred[:, :min_length] if len(pred.shape) > 1 else pred[:min_length] for pred in all_preds])
    all_labels = np.array([label[:, :min_length] if len(label.shape) > 1 else label[:min_length] for label in all_labels])

    # Ensure both arrays have the same shape
    print(f"Shape of true labels: {all_labels.shape}")
    print(f"Shape of predictions: {all_preds.shape}")

    # Reshape true_labels to match the shape of predictions if necessary
    if all_labels.shape[-1] != all_preds.shape[-1]:
        all_labels = np.repeat(all_labels, all_preds.shape[-1], axis=-1)
        print(f"Reshaped true labels to: {all_labels.shape}")

    predictions = np.vstack(all_preds)
    true_labels = np.vstack(all_labels)
    mse = mean_squared_error(true_labels, predictions)
    
    # For accuracy, you need to threshold the predictions and compare to true labels
    accuracy = accuracy_score(np.round(true_labels), np.round(predictions))
    return mse, accuracy

def load_data():
    feature_sets = ['mfccs', 'spectral', 'rmse', 'zcr', 'combined']
    test_features = {key: np.load(f'./Dataset/test_{key}_features.npy') for key in feature_sets}
    test_labels = np.load('./Dataset/test_labels.npy')

    test_datasets = {key: SnoreDataset(test_features[key], test_labels) for key in feature_sets}
    test_loaders = {key: DataLoader(test_datasets[key], batch_size=16, shuffle=False) for key in feature_sets}
    return test_loaders

def load_best_model(model_name, model_path):
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path))
    return model

def main():
    test_loaders = load_data()
    
    # Load the best model
    model_name = 'AdvancedCNNAutoencoder'  # Change this to 'UNet1D' if needed
    model_path = './best_model.pth'
    best_model = load_best_model(model_name, model_path)
    
    results = []
    feature_combinations = {
        'Spectral': ['spectral'],
        'Spectral + RMS + ZCR': ['spectral', 'rmse', 'zcr'],
        'MFCCs': ['mfccs'],
        'MFCCs + Spectral + RMS + ZCR': ['combined']
    }

    for feature_set_name, features in feature_combinations.items():
        combined_features_list = []
        for feature in features:
            current_features = test_loaders[feature].dataset.features
            if len(current_features.shape) == 1:
                current_features = current_features.reshape(-1, 1)
            combined_features_list.append(current_features)
        combined_features = np.hstack(combined_features_list)

        combined_dataset = SnoreDataset(combined_features, test_loaders['mfccs'].dataset.labels)
        combined_loader = DataLoader(combined_dataset, batch_size=16, shuffle=False)
        
        mse, accuracy = evaluate_model(best_model, combined_loader)
        results.append({
            'Feature Set': feature_set_name,
            'Model': model_name,
            'MSE': mse,
            'Accuracy': accuracy
        })

    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    main()