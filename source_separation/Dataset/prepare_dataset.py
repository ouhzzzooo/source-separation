import os
import glob
import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm

def load_wav_16k_mono(filename):
    y, sr = librosa.load(filename, sr=16000, mono=True)
    y = y / np.max(np.abs(y))  # Normalization to [-1, 1]
    return y

def save_wav_16k_mono(wav, filename, sr=16000):
    sf.write(filename, wav, sr)

def process_folder(input_folder, output_folder_original, output_folder_combined):
    snore_files = glob.glob(os.path.join(input_folder, '1/*.wav'))
    noise_files = glob.glob(os.path.join(input_folder, '0/*.wav'))

    os.makedirs(os.path.join(output_folder_original, '0'), exist_ok=True)
    os.makedirs(os.path.join(output_folder_original, '1'), exist_ok=True)
    os.makedirs(output_folder_combined, exist_ok=True)

    # Process original files
    for file in snore_files:
        wav = load_wav_16k_mono(file)
        output_file = os.path.join(output_folder_original, '1', os.path.basename(file))
        save_wav_16k_mono(wav, output_file)

    for file in noise_files:
        wav = load_wav_16k_mono(file)
        output_file = os.path.join(output_folder_original, '0', os.path.basename(file))
        save_wav_16k_mono(wav, output_file)

    # Process combined files
    for snore_file in tqdm(snore_files, desc="Combining Snore and Noise Files"):
        snore = load_wav_16k_mono(snore_file)
        noise_file = np.random.choice(noise_files)
        noise = load_wav_16k_mono(noise_file)

        if len(noise) < len(snore):
            noise = np.tile(noise, int(np.ceil(len(snore) / len(noise))))
        noise = noise[:len(snore)]

        combined = snore + noise
        combined = combined / np.max(np.abs(combined))

        output_file = os.path.join(output_folder_combined, os.path.basename(snore_file))
        save_wav_16k_mono(combined, output_file)
        print(f"save to : {output_file}")

if __name__ == "__main__":
    dataset_base = "./Dataset"
    wav_base = "./WAV"

    for split in ["train", "val", "test"]:
        input_folder = os.path.join(wav_base, split)
        output_folder_original = os.path.join(dataset_base, split.capitalize(), "original")
        output_folder_combined = os.path.join(dataset_base, split.capitalize(), "combined")
        
        process_folder(input_folder, output_folder_original, output_folder_combined)