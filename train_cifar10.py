import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# 1. Load CIFAR-10 dataset
print("📥 Loading CIFAR-10 dataset...")
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Build CNN model
print("🛠️ Building model...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# 3. Compile model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 4. Train model (10 epochs)
print("🚀 Training model for 10 epochs...")
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 5. Save model
model.save("cifar10_cnn_model.h5")
print("✅ Model trained and saved as cifar10_cnn_model.h5")
