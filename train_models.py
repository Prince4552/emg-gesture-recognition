import os
import glob
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import warnings
warnings.filterwarnings("ignore")

# Import from your custom modules
from feature_cleaning import extract_features_from_signal

# Constants
DATA_DIR = 'data/'
MODEL_DIR = 'models/'
SIGNAL_CHUNK_SIZE = 333
N_CLUSTERS = 3
RANDOM_STATE = 42
TEST_SIZE = 0.3
NOISE_LEVEL = 0.05
N_AUGMENTATIONS = 1

# Label mappings
LABEL_MAPPING = {'open': "open", 'punch': "punch", 'repos': "rest"}
FATIGUE_LABELS = ['en_forme', 'legere_fatigue', 'fatigue']


def augment_data(features, noise_level=NOISE_LEVEL, n_aug=N_AUGMENTATIONS):
    """Augment data by adding Gaussian noise."""
    augmented = [features]
    for _ in range(n_aug):
        noise = np.random.randn(*features.shape) * noise_level * np.std(features, axis=0)
        augmented.append(features + noise)
    return np.vstack(augmented)


def load_emg_data(data_dir=DATA_DIR):
    """Load EMG data from .npy files - identical to original."""
    signals = []
    labels = []
    
    for gesture_dir in os.listdir(data_dir):
        full_dir = os.path.join(data_dir, gesture_dir)
        if not os.path.isdir(full_dir):
            continue
            
        for file in glob.glob(os.path.join(full_dir, '*.npy')):
            data = np.load(file).T  # Transpose to match original
            if data.shape[0] != 7:
                print(f"Skipping {file} with shape {data.shape}")
                continue
                
            # Combine channels 5 and 6 (A2 and A3) - identical to original
            new_data = np.concatenate((data[5], data[6]))
            signals.append(new_data)
            labels.append(gesture_dir)

    return signals, labels


def create_feature_dataframe(signals, labels, chunk_size=SIGNAL_CHUNK_SIZE):
    """Create DataFrame of features from raw signals - identical to original."""
    new_signals = []
    new_labels = []
    
    for label, signal in zip(labels, signals):
        # Split signal into two channels - identical to original
        signal1 = signal[:len(signal)//2]
        signal2 = signal[len(signal)//2:]
        
        # Process in chunks - identical to original
        for i in range(0, len(signal1), chunk_size):
            if i + chunk_size > len(signal1):
                i = len(signal1) - chunk_size
                
            # Extract features from chunk - identical to original
            new_signal = np.concatenate((signal1[i:i+chunk_size], signal2[i:i+chunk_size]))
            features = extract_features_from_signal(new_signal)
            
            new_signals.append(features)
            new_labels.append(label)
    
    # Create DataFrame - identical to original
    feature_names = ["rms1", "mav1", "zc1", "ssc1", "wl1", "dom_freq1", 
                     "rms2", "mav2", "zc2", "ssc2", "wl2", "dom_freq2"]
    
    df = pd.DataFrame(new_signals, columns=feature_names)
    df['label'] = new_labels
    df['label'] = df['label'].map(LABEL_MAPPING)
    
    return df


def add_fatigue_clusters(df, n_clusters=N_CLUSTERS):
    """Add fatigue clusters to DataFrame using KMeans - identical to original."""
    # Select features for clustering
    features = df.drop(columns=['label'])
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    df['cluster'] = kmeans.fit_predict(features)
    
    return df


def get_cluster_mapping(df):
    """Map clusters to fatigue levels based on temporal ordering - identical to original."""
    # Calculate the length of each third
    third_len = len(df) // 3
    first_third = df.iloc[:third_len]
    second_third = df.iloc[third_len:2*third_len]
    third_third = df.iloc[2*third_len:]

    def get_ordered_clusters(partition, all_clusters=range(3)):
        cluster_counts = partition['cluster'].value_counts()

        for cluster in all_clusters:
            if cluster not in cluster_counts:
                cluster_counts[cluster] = 0

        return cluster_counts.sort_values(ascending=False).index.tolist()

    # Get the order of clusters for each third
    first_order = get_ordered_clusters(first_third)
    second_order = get_ordered_clusters(second_third)
    third_order = get_ordered_clusters(third_third)

    assigned_clusters = set()
    cluster_labels = {}

    # Assign en_forme from the first third's most frequent cluster not yet assigned
    for cluster in first_order:
        if cluster not in assigned_clusters:
            cluster_labels['en_forme'] = cluster
            assigned_clusters.add(cluster)
            break

    # Assign fatigue from the third third's most frequent remaining cluster
    for cluster in third_order:
        if cluster not in assigned_clusters:
            cluster_labels['fatigue'] = cluster
            assigned_clusters.add(cluster)
            break

    # Assign legere_fatigue from the second third's most frequent remaining cluster
    for cluster in second_order:
        if cluster not in assigned_clusters:
            cluster_labels['legere_fatigue'] = cluster
            assigned_clusters.add(cluster)
            break

    # Invert the dictionary to map cluster number to label
    return {v: k for k, v in cluster_labels.items()}


def train_classifier(X, y, scaler=None, model_name="", save_scaler_name=None):
    """Train a classifier and return the model and scaler."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Scale features if requested
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        if save_scaler_name:
            joblib.dump(scaler, os.path.join(MODEL_DIR, f"{save_scaler_name}_scaler.pkl"))
    
    # Train classifier
    classifier = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy for {model_name}: {accuracy * 100:.2f}%")
    
    return classifier


def train_models(use_scaler=True, use_augmentation=False):
    """Main function to train all models - functionality identical to original."""
    print("Training models...")
    
    # Create model directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load and preprocess data
    print("Reading data...")
    signals, labels = load_emg_data()
    
    print("Transforming data...")
    df = create_feature_dataframe(signals, labels)
    
    # Data augmentation - identical to original
    if use_augmentation:
        X = df.drop(columns=['label']).values
        y = df['label'].values

        X_aug = augment_data(X, noise_level=0.05, n_aug=1)
        y_aug = np.repeat(y, 1 + 1)

        df = pd.DataFrame(X_aug, columns=df.drop(columns=['label']).columns)
        df['label'] = y_aug

    # Separate data by gesture - identical to original
    open_df = df[df['label'] == 'open'].copy()
    punch_df = df[df['label'] == 'punch'].copy()

    print('Clustering data for tiredness...')
    # Clustering 'punch' data to get tiredness levels - identical to original
    punch_df = add_fatigue_clusters(punch_df, 3)
    punch_cluster_mapping = get_cluster_mapping(punch_df)
    print(punch_cluster_mapping)
    punch_df['cluster'] = punch_df['cluster'].map(punch_cluster_mapping)
    
    # Clustering 'open' data to get tiredness levels - identical to original
    open_df = add_fatigue_clusters(open_df, 3)
    open_cluster_mapping = get_cluster_mapping(open_df)
    print(open_cluster_mapping)
    open_df['cluster'] = open_df['cluster'].map(open_cluster_mapping)

    print("Training classifiers...")
    # Movement classifier - identical to original
    movement_classifier = train_classifier(
        df.drop(columns=['label']), 
        df['label'], 
        StandardScaler() if use_scaler else None,
        "Movement classifier",
        "movement"
    )
    
    # Open tiredness classifier - identical to original
    open_classifier = train_classifier(
        open_df.drop(columns=['label', 'cluster']),
        open_df['cluster'],
        StandardScaler() if use_scaler else None,
        "Open classifier",
        "open"
    )
    
    # Punch tiredness classifier - identical to original
    punch_classifier = train_classifier(
        punch_df.drop(columns=['label', 'cluster']),
        punch_df['cluster'],
        StandardScaler() if use_scaler else None,
        "Punch classifier",
        "punch"
    )
    
    # Save models - identical to original
    with open(os.path.join(MODEL_DIR, 'movement_classifier_model.pkl'), 'wb') as file:
        pickle.dump(movement_classifier, file)
    
    with open(os.path.join(MODEL_DIR, 'open_classifier_model.pkl'), 'wb') as file:
        pickle.dump(open_classifier, file)
    
    with open(os.path.join(MODEL_DIR, 'punch_classifier_model.pkl'), 'wb') as file:
        pickle.dump(punch_classifier, file)
    
    print("Models saved to '/models'")


if __name__ == "__main__":
    train_models(use_scaler=True, use_augmentation=False)