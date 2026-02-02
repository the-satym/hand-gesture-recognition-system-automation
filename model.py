import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

print("=" * 50)
print("Dynamic Gesture Model Training")
print("=" * 50)

# 1. Load the sequence data
print("\n[1/6] Loading data...")
try:
    sequences = np.load('gesture_sequences.npy')
    labels = np.load('gesture_labels.npy')
    print(f"✓ Loaded {len(sequences)} sequences")
    print(f"✓ Data shape: {sequences.shape}")  # (num_samples, sequence_length, 63)
except FileNotFoundError:
    print("Error: Data files not found!")
    print("Please run the data collection script first to create:")
    print("  - gesture_sequences.npy")
    print("  - gesture_labels.npy")
    exit()

# 2. Encode labels to numbers
print("\n[2/6] Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
y_categorical = to_categorical(y_encoded)

unique_gestures = label_encoder.classes_
num_classes = len(unique_gestures)

print(f"✓ Found {num_classes} gesture classes:")
for i, gesture in enumerate(unique_gestures):
    count = np.sum(labels == gesture)
    print(f"  - {gesture}: {count} samples")

# 3. Split data into training and testing sets
print("\n[3/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    sequences, y_categorical, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples: {len(X_test)}")

# 4. Build LSTM model
print("\n[4/6] Building LSTM model...")

model = Sequential([
    # First Bidirectional LSTM layer
    Bidirectional(LSTM(128, return_sequences=True, activation='relu'),
                  input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.3),

    # Second LSTM layer
    Bidirectional(LSTM(64, activation='relu')),
    Dropout(0.3),

    # Dense layers
    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(32, activation='relu'),

    # Output layer
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model architecture:")
model.summary()

# 5. Train the model
print("\n[5/6] Training model...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_gesture_model.h5', monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)

# 6. Evaluate the model
print("\n[6/6] Evaluating model...")

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\n{'=' * 50}")
print(f"Training Accuracy:   {train_acc * 100:.2f}%")
print(f"Testing Accuracy:    {test_acc * 100:.2f}%")
print(f"{'=' * 50}")

# 7. Save everything
print("\nSaving model and encoder...")

model.save('gesture_model.h5')
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("✓ Model saved as 'gesture_model.h5'")
print("✓ Label encoder saved as 'label_encoder.pkl'")

# 8. Test predictions on a few samples
print("\n" + "=" * 50)
print("Sample Predictions:")
print("=" * 50)

test_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)

for idx in test_indices:
    prediction = model.predict(X_test[idx:idx + 1], verbose=0)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    actual_class = label_encoder.inverse_transform([np.argmax(y_test[idx])])
    confidence = np.max(prediction) * 100

    status = "✓" if predicted_class[0] == actual_class[0] else "✗"
    print(
        f"{status} Predicted: {predicted_class[0]:15s} | Actual: {actual_class[0]:15s} | Confidence: {confidence:.1f}%")

print("\n🎉 Training complete!")