# ============================================================
#   Handwritten Digit Recognizer using CNN + MNIST Dataset
#   Internship Project - Codec Technologies
#   Author: Harshitha T N
# ============================================================

# ── STEP 1: Install required libraries (run this in terminal) ──
# pip install tensorflow numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────
# STEP 2: Load and Explore the MNIST Dataset
# ─────────────────────────────────────────
print("=" * 50)
print("  Handwritten Digit Recognizer (CNN + MNIST)")
print("=" * 50)

# Load dataset (automatically downloaded from Keras)
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"\n📦 Dataset Loaded Successfully!")
print(f"   Training samples : {X_train.shape[0]}")
print(f"   Testing  samples : {X_test.shape[0]}")
print(f"   Image shape      : {X_train.shape[1:]} (28x28 pixels)")
print(f"   Classes          : {np.unique(y_train)} (digits 0-9)")

# ─────────────────────────────────────────
# STEP 3: Visualize Sample Images
# ─────────────────────────────────────────
plt.figure(figsize=(12, 4))
plt.suptitle("Sample Images from MNIST Dataset", fontsize=14, fontweight='bold')
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}", fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.savefig("sample_digits.png", dpi=100)
plt.show()
print("\n✅ Sample digit images saved as 'sample_digits.png'")

# ─────────────────────────────────────────
# STEP 4: Preprocess the Data
# ─────────────────────────────────────────
# Normalize pixel values from [0, 255] → [0.0, 1.0]
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32")  / 255.0

# Reshape: add channel dimension (required for CNN)
# Shape: (samples, 28, 28) → (samples, 28, 28, 1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# One-hot encode labels (e.g., 3 → [0,0,0,1,0,0,0,0,0,0])
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat  = keras.utils.to_categorical(y_test,  10)

print(f"\n✅ Data Preprocessed:")
print(f"   X_train shape : {X_train.shape}")
print(f"   X_test  shape : {X_test.shape}")

# ─────────────────────────────────────────
# STEP 5: Build the CNN Model
# ─────────────────────────────────────────
model = keras.Sequential([

    # --- Convolutional Block 1 ---
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1),
                  padding='same', name='conv1'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # --- Convolutional Block 2 ---
    layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # --- Fully Connected Layers ---
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')   # 10 output classes (0–9)
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📐 CNN Model Architecture:")
model.summary()

# ─────────────────────────────────────────
# STEP 6: Train the Model
# ─────────────────────────────────────────
print("\n🚀 Training the model...")

# Stop early if validation accuracy stops improving
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=3, restore_best_weights=True
)

history = model.fit(
    X_train, y_train_cat,
    epochs=15,
    batch_size=128,
    validation_split=0.1,    # 10% of training data used for validation
    callbacks=[early_stop],
    verbose=1
)

# ─────────────────────────────────────────
# STEP 7: Evaluate the Model
# ─────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n🎯 Test Accuracy : {test_acc * 100:.2f}%")
print(f"   Test Loss     : {test_loss:.4f}")

# ─────────────────────────────────────────
# STEP 8: Plot Training History
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training History", fontsize=14, fontweight='bold')

# Accuracy plot
axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='steelblue')
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='orange')
axes[0].set_title("Model Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

# Loss plot
axes[1].plot(history.history['loss'],     label='Train Loss', color='steelblue')
axes[1].plot(history.history['val_loss'], label='Val Loss',   color='orange')
axes[1].set_title("Model Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("training_history.png", dpi=100)
plt.show()
print("✅ Training history saved as 'training_history.png'")

# ─────────────────────────────────────────
# STEP 9: Confusion Matrix
# ─────────────────────────────────────────
y_pred_probs = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.show()
print("✅ Confusion matrix saved as 'confusion_matrix.png'")

# ─────────────────────────────────────────
# STEP 10: Classification Report
# ─────────────────────────────────────────
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))

# ─────────────────────────────────────────
# STEP 11: Predict on Sample Test Images
# ─────────────────────────────────────────
plt.figure(figsize=(14, 5))
plt.suptitle("Model Predictions on Test Images", fontsize=14, fontweight='bold')

for i in range(10):
    idx        = np.random.randint(0, len(X_test))
    img        = X_test[idx]
    true_label = y_test[idx]
    pred_label = y_pred[idx]
    confidence = y_pred_probs[idx][pred_label] * 100

    plt.subplot(2, 5, i + 1)
    plt.imshow(img.reshape(28, 28), cmap='gray')
    color = 'green' if pred_label == true_label else 'red'
    plt.title(f"True:{true_label}  Pred:{pred_label}\n{confidence:.1f}%",
              color=color, fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.savefig("predictions.png", dpi=100)
plt.show()
print("✅ Sample predictions saved as 'predictions.png'")

# ─────────────────────────────────────────
# STEP 12: Save the Model
# ─────────────────────────────────────────
model.save("digit_recognizer_model.h5")
print("\n💾 Model saved as 'digit_recognizer_model.h5'")

print("\n" + "=" * 50)
print("  ✅ Project 1 Complete!")
print(f"  Final Test Accuracy: {test_acc * 100:.2f}%")
print("=" * 50)
