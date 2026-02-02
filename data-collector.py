import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from collections import deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

# Configuration for dynamic gestures
SEQUENCE_LENGTH = 30  # Number of frames per gesture (adjust based on gesture speed)
sequences = []
labels = []

gesture_name = input("Enter gesture name (e.g., swipe_left, circle, wave): ")

print("\n=== Dynamic Gesture Collection ===")
print("Press 'r' to START recording a gesture sequence")
print("Press 's' to STOP recording (or auto-stops after 30 frames)")
print("Press 'q' to quit and save")
print("=" * 40)

recording = False
current_sequence = []
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Status display
    if recording:
        cv2.putText(frame, f"RECORDING: {frame_count}/{SEQUENCE_LENGTH}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
    else:
        cv2.putText(frame, "Press 'r' to record",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Samples: {len(sequences)}",
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            if recording:
                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

                current_sequence.append(landmarks)
                frame_count += 1

                # Auto-stop when sequence is complete
                if frame_count >= SEQUENCE_LENGTH:
                    sequences.append(current_sequence.copy())
                    labels.append(gesture_name)
                    print(f"✓ Captured sequence {len(sequences)}")

                    # Reset
                    recording = False
                    current_sequence = []
                    frame_count = 0

    cv2.imshow("Dynamic Gesture Collection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r') and not recording:
        if results.multi_hand_landmarks:
            recording = True
            current_sequence = []
            frame_count = 0
            print("Started recording...")
        else:
            print("No hand detected! Show your hand to the camera.")

    elif key == ord('s') and recording:
        if frame_count >= 10:  # Minimum 10 frames
            sequences.append(current_sequence.copy())
            labels.append(gesture_name)
            print(f"✓ Captured sequence {len(sequences)} (early stop at {frame_count} frames)")
        else:
            print(f"✗ Sequence too short ({frame_count} frames), need at least 10")

        recording = False
        current_sequence = []
        frame_count = 0

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save sequences to file
if len(sequences) > 0:
    # Pad sequences to same length if needed
    max_len = max(len(seq) for seq in sequences)

    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_len:
            # Pad with zeros
            padding = [seq[-1]] * (max_len - len(seq))  # Repeat last frame
            seq = seq + padding
        padded_sequences.append(seq)

    # Convert to numpy array
    data_array = np.array(padded_sequences)
    labels_array = np.array(labels)

    print(f"\nData shape: {data_array.shape}")  # Should be (num_samples, sequence_length, 63)

    # Save as .npy files (better for sequence data)
    npy_file = 'gesture_sequences.npy'
    labels_file = 'gesture_labels.npy'

    if os.path.isfile(npy_file):
        # Append to existing data
        existing_data = np.load(npy_file)
        existing_labels = np.load(labels_file)

        data_array = np.vstack([existing_data, data_array])
        labels_array = np.concatenate([existing_labels, labels_array])

    np.save(npy_file, data_array)
    np.save(labels_file, labels_array)

    print(f"✓ Saved {len(sequences)} sequences to {npy_file}")
    print(f"✓ Total sequences in dataset: {len(data_array)}")
else:
    print("No data collected!")