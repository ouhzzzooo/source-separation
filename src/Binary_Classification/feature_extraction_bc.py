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
    return features

def reshape_features(features, feature_set):
    # Ensure the reshaping matches the expected input of the conv layers
    if feature_set == 'mfccs':
        return features[:13].reshape(1, 13, -1)
    elif feature_set == 'spectral':
        return features[13:16].reshape(1, 3, -1)
    elif feature_set == 'rmse':
        return features[16:17].reshape(1, 1, -1)
    elif feature_set == 'zcr':
        return features[17:18].reshape(1, 1, -1)
    elif feature_set == 'combined':
        return features.reshape(1, 18, -1)
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")