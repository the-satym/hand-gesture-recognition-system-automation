import cv2
import mediapipe as mp
import numpy as np
import pickle


import pyautogui
import time
from collections import deque

print("=" * 50)
print("Dynamic Gesture Recognition System")
print("=" * 50)

# Load the trained model and label encoder
print("\n[1/3] Loading model...")
try:
    model = load_model('gesture_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("✓ Model loaded successfully")
except FileNotFoundError:
    print("Error: Model files not found!")
    print("Please train the model first using the training script.")
    exit()

# Initialize MediaPipe
print("\n[2/3] Initializing camera...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Configuration
SEQUENCE_LENGTH = 30 # Must match training data
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to trigger action
COOLDOWN = 1.5  # Seconds between actions

# State variables
sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
last_action_time = 0
current_gesture = "None"
confidence = 0.0

print("\n[3/3] System Ready!")
print("=" * 50)
print("Gesture Controls:")
print("  - Palm: Play/Pause (Space)")
print("  - Index: Mute/Unmute")
print("  - Right Slide: Scroll Up")
print("  - Left Slide: Scroll Down")
print("\nPress 'q' to quit")
print("=" * 50)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Extract landmarks
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Add to sequence buffer
            sequence_buffer.append(landmarks)

            # Only predict when buffer is full
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                # Prepare input for model
                input_sequence = np.array(list(sequence_buffer))
                input_sequence = np.expand_dims(input_sequence, axis=0)  # Shape: (1, 30, 63)

                # Make prediction
                prediction = model.predict(input_sequence, verbose=0)
                predicted_index = np.argmax(prediction)
                confidence = np.max(prediction)

                # Only update gesture if confidence is high enough
                if confidence >= CONFIDENCE_THRESHOLD:
                    current_gesture = label_encoder.inverse_transform([predicted_index])[0]

                    # Execute actions with cooldown
                    current_time = time.time()
                    if current_time - last_action_time > COOLDOWN:

                        if current_gesture == "plam":
                            pyautogui.press('space')
                            last_action_time = current_time
                            print(f"✓ Action: Play/Pause (Confidence: {confidence * 100:.1f}%)")

                        elif current_gesture == "peace":
                            pyautogui.press('volumemute')
                            last_action_time = current_time
                            print(f"✓ Action: Mute/Unmute (Confidence: {confidence * 100:.1f}%)")

                        elif current_gesture == "right slide":
                            pyautogui.scroll(-300)
                            last_action_time = current_time
                            print(f"✓ Action: Scroll Up (Confidence: {confidence * 100:.1f}%)")

                        elif current_gesture == "left slide":
                            pyautogui.scroll(300)
                            last_action_time = current_time
                            print(f"✓ Action: Scroll Down (Confidence: {confidence * 100:.1f}%)")

                else:
                    current_gesture = "Uncertain"

    else:
        # No hand detected - clear buffer
        sequence_buffer.clear()
        current_gesture = "No Hand"
        confidence = 0.0

    # Display information on frame
    # Status bar background
    cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)

    # Gesture name
    cv2.putText(frame, f"Gesture: {current_gesture}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Confidence bar
    if confidence > 0:
        cv2.putText(frame, f"Confidence: {confidence * 100:.1f}%", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Confidence bar
        bar_width = int(confidence * 300)
        color = (0, 255, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        cv2.rectangle(frame, (10, 80), (10 + bar_width, 95), color, -1)

    # Buffer status
    buffer_status = f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}"
    cv2.putText(frame, buffer_status, (w - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Show frame
    cv2.imshow("Dynamic Gesture Control", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nShutting down...")
        break

cap.release()
cv2.destroyAllWindows()
print("✓ System closed")