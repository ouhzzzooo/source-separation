import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
from tqdm import tqdm
from model_reconstructed import get_model, EarlyStopping, combined_loss

class SnoreDataset(Dataset):
    def __init__(self, combined_path, clean_path):
        self.combined_files = glob.glob(os.path.join(combined_path, '*.wav'))
        self.clean_files = {os.path.basename(f): f for f in glob.glob(os.path.join(clean_path, '1/*.wav'))}
        self.filtered_combined_files = self.filter_files(self.combined_files)
        print(f"Found {len(self.combined_files)} combined files in {combined_path}")
        print(f"Filtered {len(self.filtered_combined_files)} combined files with valid amplitudes")
        print(f"Found {len(self.clean_files)} clean files in {clean_path}/1")

    def __len__(self):
        return len(self.filtered_combined_files)

    def __getitem__(self, idx):
        combined_file = self.filtered_combined_files[idx]
        combined_wav, _ = librosa.load(combined_file, sr=16000, mono=True)
        combined_wav = combined_wav / np.max(np.abs(combined_wav))

        file_name = os.path.basename(combined_file)
        clean_file = self.clean_files.get(file_name, None)
        clean_wav = np.zeros_like(combined_wav)
        if clean_file:
            clean_wav, _ = librosa.load(clean_file, sr=16000, mono=True)
            clean_wav = clean_wav / np.max(np.abs(clean_wav))

        return torch.tensor(combined_wav, dtype=torch.float32).unsqueeze(0), torch.tensor(clean_wav, dtype=torch.float32).unsqueeze(0)

    def filter_files(self, file_list):
        filtered_files = []
        for file in file_list:
            wav, _ = librosa.load(file, sr=16000, mono=True)
            if np.max(np.abs(wav)) > 0:
                filtered_files.append(file)
            else:
                print(f"Skipping file {file} with max amplitude 0.")
        return filtered_files

def train_model(model, train_loader, val_loader, num_epochs, patience, model_path):
    criterion = combined_loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4, verbose=True)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    finish = False
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (combined, clean) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            combined, clean = combined.cuda(), clean.cuda()
            optimizer.zero_grad()
            output = model(combined)

            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"NaN or Inf in model output at batch {batch_idx}")
                continue

            loss = criterion(output, clean)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf in loss at batch {batch_idx}")
                continue

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for combined, clean in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                combined, clean = combined.cuda(), clean.cuda()
                output = model(combined)

                if torch.isnan(output).any() or torch.isinf(output).any():
                    print(f"NaN or Inf in model output during validation at epoch {epoch+1}")
                    continue

                loss = criterion(output, clean)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN or Inf in validation loss at epoch {epoch+1}")
                    continue

                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
             

if __name__ == "__main__":
    train_combined_path = "./src/Dataset/Train/combined"
    val_combined_path = "./src/Dataset/Val/combined"
    train_clean_path = "./src/Dataset/Train/original"
    val_clean_path = "./src/Dataset/Val/original"
    model_path = "./src/Reconstructed/reconstructed_model_UNet_2.pth"

    train_dataset = SnoreDataset(train_combined_path, train_clean_path)
    val_dataset = SnoreDataset(val_combined_path, val_clean_path)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("No data found. Please check the dataset paths and ensure there are .wav files in the combined directories.")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model_name = 'UNet1D'  
    model = get_model(model_name).cuda()
    train_model(model, train_loader, val_loader, num_epochs=100, patience=10, model_path=model_path)