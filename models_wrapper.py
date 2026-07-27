import pandas as pd
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Optional


class ModelsWrapper:
    def __init__(self, gestures: list, use_scaler: bool = True, models_dir: str = "models"):
        self.classifiers: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.use_scaler = use_scaler
        self.models_dir = Path(models_dir)

        # Ensure models directory exists
        self.models_dir.mkdir(exist_ok=True)

        for gesture in gestures:
            self.load_models(gesture)

    def load_models(self, name: str) -> None:
        """Load both classifier and scaler for a gesture"""
        self.load_classifier(name)
        if self.use_scaler:
            self.load_scaler(name)

    def load_classifier(self, name: str) -> None:
        """Load a classifier model from file"""
        model_path = self.models_dir / f"{name}_classifier_model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Classifier model not found: {model_path}")

        try:
            with open(model_path, 'rb') as file:
                self.classifiers[name] = pickle.load(file)
        except Exception as e:
            raise IOError(f"Failed to load classifier {name}: {e}")

    def load_scaler(self, name: str) -> None:
        """Load a scaler from file"""
        scaler_path = self.models_dir / f"{name}_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        try:
            self.scalers[name] = joblib.load(scaler_path)
        except Exception as e:
            raise IOError(f"Failed to load scaler {name}: {e}")

    def get_classifier(self, name: str) -> Any:
        """Get classifier for a gesture"""
        classifier = self.classifiers.get(name)
        if classifier is None:
            raise ValueError(f"No classifier found for gesture: {name}")
        return classifier

    def get_scaler(self, name: str) -> Any:
        """Get scaler for a gesture"""
        scaler = self.scalers.get(name)
        if scaler is None:
            raise ValueError(f"No scaler found for gesture: {name}")
        return scaler

    def get_prediction(self, name: str, features_df: pd.DataFrame) -> Any:
        """Get prediction for features using the specified gesture model"""
        if name not in self.classifiers:
            raise ValueError(f"No model loaded for gesture: {name}")

        features_df_copy = features_df.copy()

        if self.use_scaler:
            scaler = self.get_scaler(name)
            features_df_copy = self._scale_features(features_df_copy, scaler)

        classifier = self.get_classifier(name)
        return classifier.predict(features_df_copy)[0]

    def _scale_features(self, features_df: pd.DataFrame, scaler: Any) -> pd.DataFrame:
        """Scale features using the provided scaler"""
        features_scaled = scaler.transform(features_df)
        return pd.DataFrame(features_scaled, columns=features_df.columns)
