from collect_data import save_gestures
from train_models import train_models
import logging
from config.settings import USE_SCALER, USE_AUGMENTATION

logger = logging.getLogger(__name__)

def calibrate(device):
    """
    Perform full calibration pipeline: data collection followed by model training.
    
    Args:
        device: The EMG device instance for data acquisition
    
    Returns:
        bool: True if calibration succeeded, False otherwise
    """
    try:
        logger.info("Starting calibration process...")
        
        # Step 1: Collect gesture data
        logger.info("Collecting gesture data...")
        save_gestures(device)
        
        # Step 2: Train models on collected data
        logger.info("Training models...")
        train_models(use_scaler=USE_SCALER, use_augmentation=USE_AUGMENTATION)
        
        logger.info("Calibration completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        return False