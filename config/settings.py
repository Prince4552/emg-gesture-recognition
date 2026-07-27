# environment setup
MODE = "development"  # "production" or "development"
GESTURE_NAMES = ["movement", "open", "punch"]
FATIGUE_GESTURES = ["open", "punch"]
USE_SCALER = True
USE_AUGMENTATION = True
SAMPLES_PER_READ = 330
CHANNEL_A2 = 5  # EMG channel A2
CHANNEL_A3 = 6  # EMG channel A3
FEATURE_NAMES = ["rms1", "mav1", "zc1", "ssc1",
                 "wl1", "dom_freq1", "rms2", "mav2", "zc2", "ssc2", "wl2", "dom_freq2"]
LOOP_DELAY = 0.1

# BITAlino setup
BITALINO_MACADDRESS = "20:19:07:00:7F:7A"
SAMPLING_RATE = 1000
ACQ_CHANNELS = [1, 2]
WINDOW_DURATION = 1  # (s)
N_SAMPLES = 100
FS = 1000
WINDOW_SIZE = 256
STEP_SIZE = 128

# Unity connection setup
UNITY_IP = "127.0.0.1"
UNITY_PORT = 6064
