import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score
from torch.utils.data import DataLoader, Dataset
from model_reconstructed import UNet1D, AdvancedCNNAutoencoder

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
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    predictions = np.vstack(all_preds)
    true_labels = np.vstack(all_labels)
    mse = mean_squared_error(true_labels, predictions)
    
    # For accuracy, you need to threshold the predictions and compare to true labels
    accuracy = accuracy_score(np.round(true_labels), np.round(predictions))
    return mse, accuracy

def load_data():
    feature_sets = ['mfccs', 'spectral', 'rmse', 'zcr', 'combined']
    test_features = {key: np.load(f'../Dataset/test_{key}_features.npy') for key in feature_sets}
    test_labels = np.load('../Dataset/test_labels.npy')

    test_datasets = {key: SnoreDataset(test_features[key], test_labels) for key in feature_sets}
    test_loaders = {key: DataLoader(test_datasets[key], batch_size=16, shuffle=False) for key in feature_sets}
    return test_loaders

def load_models():
    unet_model = UNet1D()
    unet_model.load_state_dict(torch.load('../Reconstructed/unet1d_model.pth'))

    cnn_autoencoder_model = AdvancedCNNAutoencoder()
    cnn_autoencoder_model.load_state_dict(torch.load('../Reconstructed/advanced_cnn_autoencoder_model.pth'))
    
    return unet_model, cnn_autoencoder_model

def main():
    test_loaders = load_data()
    unet_model, cnn_autoencoder_model = load_models()
    
    results = []
    feature_combinations = {
        'Spectral': ['spectral'],
        'Spectral + RMS + ZCR': ['spectral', 'rmse', 'zcr'],
        'MFCCs': ['mfccs'],
        'MFCCs + Spectral + RMS + ZCR': ['combined']
    }

    for feature_set_name, features in feature_combinations.items():
        combined_features = np.hstack([test_loaders[feature].dataset.features for feature in features])
        combined_dataset = SnoreDataset(combined_features, test_loaders['mfccs'].dataset.labels)
        combined_loader = DataLoader(combined_dataset, batch_size=16, shuffle=False)
        
        unet_mse, unet_accuracy = evaluate_model(unet_model, combined_loader)
        cnn_autoencoder_mse, cnn_autoencoder_accuracy = evaluate_model(cnn_autoencoder_model, combined_loader)
        results.append({
            'Feature Set': feature_set_name,
            'Model': 'UNet1D',
            'MSE': unet_mse,
            'Accuracy': unet_accuracy
        })
        results.append({
            'Feature Set': feature_set_name,
            'Model': 'AdvancedCNNAutoencoder',
            'MSE': cnn_autoencoder_mse,
            'Accuracy': cnn_autoencoder_accuracy
        })

    df = pd.DataFrame(results)
    print(df)

if __name__ == "__main__":
    main()