import numpy as np
import time
import pandas as pd
from device import Device
from bitalino import BITalino
from calibrate import calibrate
from models_wrapper import ModelsWrapper
from socket_wrapper import SocketWrapper
from feature_cleaning import extract_features_from_signal
from config.settings import *
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_system():
    """Initialize all system components"""
    try:
        # UDP setup
        sock = SocketWrapper(UNITY_IP, UNITY_PORT)
        
        # Device setup
        device = Device(BITALINO_MACADDRESS, SAMPLING_RATE, ACQ_CHANNELS)
        device.start()
        
        # Check if models exist, calibrate if not
        if not os.path.exists('models'):
            logger.info("No models found. Starting calibration...")
            calibrate(device)
            device.reset()  # Reset after calibration
        
        # Load models
        models = ModelsWrapper(GESTURE_NAMES, USE_SCALER)
        
        return sock, device, models
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

def process_emg_data(device, models):
    """Process a single batch of EMG data"""
    try:
        # Read raw data
        raw_data = device.read(SAMPLES_PER_READ)
        
        # Extract and process signals
        data = np.array(raw_data)
        a2_signal = data[:, CHANNEL_A2].astype(float)
        a3_signal = data[:, CHANNEL_A3].astype(float)
        combined_signal = np.concatenate((a2_signal, a3_signal))
        
        # Extract features
        features = extract_features_from_signal(
            combined_signal, FS, WINDOW_SIZE, STEP_SIZE)
        
        # Create features DataFrame
        features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        
        return features_df
    except Exception as e:
        logger.error(f"Error processing EMG data: {e}")
        return None

def predict_gesture_and_fatigue(models, features_df):
    """Predict gesture and fatigue level from features"""
    if features_df is None:
        return None, None
        
    try:
        # Predict movement
        predicted_movement = models.get_prediction("movement", features_df)
        logger.info(f"Geste détecté : {predicted_movement.upper()}")
        
        # Predict fatigue level for relevant gestures
        predicted_fatigue = None
        if predicted_movement in FATIGUE_GESTURES:
            predicted_fatigue = models.get_prediction(predicted_movement, features_df)
        else:
            logger.info(f"Pas de prédiction de fatigue pour ce geste: {predicted_movement.upper()}")
            
        return predicted_movement, predicted_fatigue
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None, None

def main():
    """Main application loop"""
    sock, device, models = None, None, None
    
    try:
        # Initialize system components
        sock, device, models = initialize_system()
        
        logger.info("Starting EMG gesture recognition...")
        
        # Main processing loop
        while True:
            # Process EMG data
            features_df = process_emg_data(device, models)
            
            # Make predictions
            movement, fatigue = predict_gesture_and_fatigue(models, features_df)
            
            # Send data to Unity if we have a valid prediction
            if movement is not None:
                data = f"{movement},{fatigue if fatigue is not None else 'N/A'}"
                sock.send(data)
                logger.debug(f"Sent to Unity: {data}")
            
            # Control loop rate
            time.sleep(LOOP_DELAY)
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Clean up resources
        logger.info("Cleaning up resources...")
        if device:
            try:
                device.stop()
                device.close()
            except Exception as e:
                logger.error(f"Error closing device: {e}")
        logger.info("Application stopped")

if __name__ == "__main__":
    main()