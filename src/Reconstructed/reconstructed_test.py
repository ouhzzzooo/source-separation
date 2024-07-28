import os
import glob
import numpy as np
import torch
import librosa
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
from model_reconstructed import get_model
from tqdm import tqdm

class SnoreDataset(Dataset):
    def __init__(self, data_path):
        self.files = glob.glob(os.path.join(data_path, '*.wav'))
        self.filtered_files = self.filter_files(self.files)
        print(f"Found {len(self.files)} files in {data_path}")
        print(f"Filtered {len(self.filtered_files)} files with valid amplitudes")

    def __len__(self):
        return len(self.filtered_files)

    def __getitem__(self, idx):
        file = self.filtered_files[idx]
        wav, _ = librosa.load(file, sr=16000, mono=True)
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            wav = wav / max_val
        return torch.tensor(wav, dtype=torch.float32), os.path.basename(file)

    def filter_files(self, file_list):
        filtered_files = []
        for file in file_list:
            wav, _ = librosa.load(file, sr=16000, mono=True)
            if np.max(np.abs(wav)) > 0:
                filtered_files.append(file)
            else:
                print(f"Skipping file {file} with max amplitude 0.")
        return filtered_files

def reconstruct_and_save(model, test_loader, output_path):
    os.makedirs(output_path, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch_idx, (combined, filenames) in enumerate(tqdm(test_loader, desc="Reconstructing Test Data")):
            combined = combined.float()
            combined = combined.unsqueeze(1)  # Ensure inputs have shape (batch_size, 1, sequence_length)
            reconstructed = model(combined)

            for i, wav in enumerate(reconstructed):
                wav = wav.squeeze().numpy()  # Remove channel dimension and convert to numpy
                max_val = np.max(np.abs(wav))
                if max_val > 0:
                    wav = wav / max_val
                else:
                    print(f"Warning: Reconstructed file {filenames[i]} has max amplitude 0.")
                sf.write(os.path.join(output_path, filenames[i]), wav, 16000)

if __name__ == "__main__":
    test_data_path = "./src/Dataset/Test/combined"
    model_path = "./src/Reconstructed/reconstructed_model_Uniform.pth"
    # "./path_try/advanced_cnn_autoencoder_model.pth"
    output_path = "./src/Reconstructed/reconstructed_data"

    test_dataset = SnoreDataset(test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model_name = 'Uniform'  # Change this to 'UNet1D' to use the other model
    model = get_model(model_name)
    model.load_state_dict(torch.load(model_path))

    reconstruct_and_save(model, test_loader, output_path)