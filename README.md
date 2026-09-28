# EMG Gesture Recognition for Unity Game Control

A real-time machine learning system that processes electromyography (EMG) signals to detect hand gestures and fatigue levels, then transmits them to a Unity game as control inputs. This project demonstrates end-to-end development of a biomedical signal processing and machine learning pipeline.

## 🚀 Key Features

- **Real-time EMG Signal Processing**: Captures and processes raw EMG data at 1000Hz sampling rate
- **Advanced Feature Extraction**: Calculates RMS, MAV, ZC, SSC, WL, and frequency-domain features
- **Machine Learning Classification**: Implements HistGradientBoosting classifiers for gesture recognition
- **Fatigue Detection**: Uses K-Means clustering to detect three levels of muscle fatigue
- **Real-time Unity Integration**: Sends classification results via UDP socket to our Unity game
- **End-to-End Pipeline**: From data acquisition to model deployment

## 🛠️ Technical Stack

### Data Acquisition & Processing
- **BITalino API**: For EMG device communication
- **NumPy & SciPy**: Signal processing and numerical computation
- **Digital Filtering**: Butterworth bandpass filter (20-450Hz)

### Machine Learning
- **Scikit-learn**: HistGradientBoosting classifiers and K-Means clustering
- **Feature Engineering**: 12 time-domain and frequency-domain features per gesture
- **Model Serialization**: Pickle and Joblib for model persistence

### System Architecture
- **Multi-threading**: Concurrent data acquisition and processing
- **UDP Socket Communication**: Real-time data transmission to Unity
- **Modular Design**: Separated data acquisition, processing, and ML components

## 📊 Machine Learning Pipeline

1. **Data Collection**: Recorded 60-second samples for each gesture (rest, fist, open hand)
2. **Signal Processing**: Bandpass filtering and normalization
3. **Feature Extraction**: 12 features per channel (RMS, MAV, ZC, SSC, WL, dominant frequency)
4. **Gesture Classification**: Multi-class classifier detecting hand gestures
5. **Fatigue Analysis**: Unsupervised clustering to detect fatigue levels
6. **Real-time Prediction**: <100ms latency from signal acquisition to classification

## 🎯 Skills Demonstrated

This project showcases expertise in:

- **Signal Processing**: EMG signal filtering, feature extraction, and noise reduction
- **Machine Learning**: Supervised classification and unsupervised clustering
- **Software Engineering**: Modular code architecture, threading, and socket programming
- **Data Pipelines**: End-to-end development from raw data to deployed model
- **Real-time Systems**: Low-latency processing and prediction
- **Biomedical Engineering**: EMG signal interpretation and fatigue analysis

## 📈 Performance Metrics

- **Gesture Recognition Accuracy**: 92-96% (varies by gesture)
- **Fatigue Level Detection**: 85-90% accuracy
- **Latency**: <100ms end-to-end processing
- **Sample Rate**: 1000Hz real-time processing

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Prince4552/emg-gesture-recognition.git
cd emg-gesture-recognition

# Install dependencies
pip install -r requirements.txt

# Connect BITalino device and run calibration
python main.py
```

## Usage

1. **Calibration**: The system will automatically prompt for calibration if no models are found
2. **Data Collection**: Perform each gesture (rest, fist, open hand) for 60 seconds when prompted
3. **Automatic Training**: Models train automatically after data collection
4. **Real-time Control**: Start the Unity application and begin gesture control

---

**Note**: This project was developed as part of an academic project in biomedical signal processing and machine learning applications for human-computer interaction.
