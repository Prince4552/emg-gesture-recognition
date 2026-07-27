import os
import time
import threading
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class GestureRecorder:
    def __init__(self, device, data_dir='data', record_duration=60, n_samples=100):
        self.device = device
        self.data_dir = data_dir
        self.record_duration = record_duration
        self.n_samples = n_samples
        self.capture_active = False
        self.data = []
        self.data_lock = threading.Lock()
        self.gestures = ['repos', 'punch', 'open']
        self.user_stopped = False

    def capture_data(self):
        """Continuously read data from device while capture is active."""
        try:
            while self.capture_active:
                try:
                    new_data = self.device.read(self.n_samples)
                    with self.data_lock:
                        self.data.extend(new_data)
                    time.sleep(0.01)  # Small sleep to prevent CPU overload
                except Exception as e:
                    logger.error(f"Error reading from device: {e}")
                    break
        except Exception as e:
            logger.error(f"Capture thread failed: {e}")

    def wait_for_enter(self):
        """Wait for user to press Enter to stop recording."""
        input()  # This will block until Enter is pressed
        self.user_stopped = True
        self.capture_active = False

    def save_gesture(self, gesture: str, target_name: str = None):
        """Save data for a specific gesture."""
        if target_name is None:
            target_name = f"gesture_{int(time.time())}"

        # Create directory if it doesn't exist
        gesture_dir = os.path.join(self.data_dir, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        # Instructions for the user
        instructions = {
            'repos': "Keep your hand in resting position",
            'punch': "Make a fist",
            'open': "Keep your hand open"
        }

        print(
            f"\n{instructions.get(gesture, 'Perform the gesture')} for up to {self.record_duration} seconds.")
        print("Press Enter to start recording...")
        input()  # Wait for user to press Enter to start

        time.sleep(1)  # Brief pause before starting

        # Reset data and flags
        with self.data_lock:
            self.data = []
        self.user_stopped = False
        self.capture_active = True

        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_data)
        capture_thread.daemon = True
        capture_thread.start()

        # Start thread to wait for Enter to stop
        stop_thread = threading.Thread(target=self.wait_for_enter)
        stop_thread.daemon = True
        stop_thread.start()

        print(f"Recording '{gesture}'... Press Enter to stop early.")
        start_time = time.time()

        # Wait for either Enter or timeout
        try:
            while (time.time() - start_time < self.record_duration and
                   self.capture_active and not self.user_stopped):
                time.sleep(0.1)  # Small sleep to prevent CPU overload

            if self.user_stopped:
                print("Recording stopped by user.")
            else:
                print("Recording completed (timeout).")

        except KeyboardInterrupt:
            print("\nRecording interrupted by user")
        finally:
            # Stop capture
            self.capture_active = False
            capture_thread.join(timeout=1.0)
            stop_thread.join(timeout=0.1)

            # Save data
            with self.data_lock:
                if self.data:
                    filename = os.path.join(gesture_dir, f"{target_name}.npy")
                    np.save(filename, np.array(self.data))
                    print(
                        f"Data saved to {filename} ({len(self.data)} samples)")
                else:
                    print("No data collected")

    def save_gestures(self):
        """Save data for all gestures."""
        for gesture in self.gestures:
            print(f"\n{'='*50}")
            print(f"Recording gesture: {gesture}")
            print(f"{'='*50}")

            try:
                self.save_gesture(gesture)
            except Exception as e:
                logger.error(f"Failed to record {gesture}: {e}")
                continue

            # Brief pause between gestures
            time.sleep(1)


def save_gestures(device):
    """Main function to save gestures data."""
    recorder = GestureRecorder(device)
    recorder.save_gestures()
