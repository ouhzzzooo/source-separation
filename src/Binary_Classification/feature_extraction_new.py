import numpy as np
import librosa

def extract_features(audio_path, sr=16000):
    y, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    
    # Concatenate all features along the feature dimension
    features = np.concatenate((mfccs, spectral_centroid, rms, zcr), axis=0)
    # Ensure the output has a single channel dimension for Conv1D input
    return features

def reshape_features(features):
    # Reshape to match the expected shape: (num_channels, sequence_length)
    return features.reshape(16, -1)

# Example usage
# features = extract_features('path/to/audio.wav')
# reshaped_features = reshape_features(features)