"""Train a convolutional sign-letter classifier from Sign Language MNIST."""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"


def main() -> None:
    train_path = DATA_DIR / "sign_mnist_train.csv"
    test_path = DATA_DIR / "sign_mnist_test.csv"
    if not train_path.exists() or not test_path.exists():
        raise SystemExit(
            "Dataset not found. Add sign_mnist_train.csv and sign_mnist_test.csv to data/."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if "label" not in train or "label" not in test:
        raise SystemExit("Both CSV files must contain a 'label' column.")
    x = train.drop(columns="label").to_numpy(dtype=np.float32).reshape(-1, 28, 28, 1) / 255.0
    y = train["label"].to_numpy(dtype=np.int32)
    x_test = test.drop(columns="label").to_numpy(dtype=np.float32).reshape(-1, 28, 28, 1) / 255.0
    y_test = test["label"].to_numpy(dtype=np.int32)
    if x.shape[1:] != (28, 28, 1) or x_test.shape[1:] != (28, 28, 1):
        raise SystemExit("Expected exactly 784 pixel columns (28 × 28) in each CSV.")

    classes = 25
    expected_labels = set(range(25)) - {9}
    if set(np.unique(y)) != expected_labels or set(np.unique(y_test)) != expected_labels:
        raise SystemExit("Expected the standard Sign Language MNIST labels 0–24, excluding 9 (J).")
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    tf.keras.utils.set_random_seed(42)
    model = Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        RandomRotation(0.04, fill_mode="constant"),
        RandomTranslation(0.06, 0.06, fill_mode="constant"),
        RandomZoom(0.08, fill_mode="constant"),
        Conv2D(32, 3, padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(32, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Dropout(0.20),
        Conv2D(64, 3, padding="same", activation="relu"),
        BatchNormalization(),
        Conv2D(64, 3, padding="same", activation="relu"),
        MaxPooling2D(),
        Dropout(0.30),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.40),
        # Preserve the source labels: output 9 is unused (J is absent).
        Dense(classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-5),
    ]
    model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=20,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_DIR / "sign_mnist_cnn.keras"
    model.save(output_path)
    print(f"Test accuracy: {accuracy:.4f} | test loss: {loss:.4f}")
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
