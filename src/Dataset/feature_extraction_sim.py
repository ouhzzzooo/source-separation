import numpy as np
import librosa

def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y=y)

    features = {
        'mfccs': np.mean(mfccs, axis=1),
        'spectral': np.hstack([np.mean(spectral_centroid), np.mean(spectral_bandwidth), np.mean(spectral_rolloff)]),
        'rmse': np.mean(rmse),
        'zcr': np.mean(zcr),
        'combined': np.hstack([np.mean(mfccs, axis=1), np.mean(spectral_centroid), np.mean(spectral_bandwidth), np.mean(spectral_rolloff), np.mean(rmse), np.mean(zcr)])
    }
    return features