"""Train the deployment MLP using the provided Sign Language MNIST CSVs."""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


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
    x = train.drop(columns="label").to_numpy(dtype=np.float32) / 255.0
    y = train["label"].to_numpy(dtype=np.int32)
    x_test = test.drop(columns="label").to_numpy(dtype=np.float32) / 255.0
    y_test = test["label"].to_numpy(dtype=np.int32)
    if x.shape[1] != 784 or x_test.shape[1] != 784:
        raise SystemExit("Expected exactly 784 pixel columns (28 × 28) in each CSV.")

    classes = 25
    expected_labels = set(range(25)) - {9}
    if set(np.unique(y)) != expected_labels:
        raise SystemExit("Expected the standard Sign Language MNIST labels 0–24, excluding 9 (J).")
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    tf.keras.utils.set_random_seed(42)
    model = Sequential([
        tf.keras.Input(shape=(784,)),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        # Keep the notebook's 25 output positions; label 9 (J) is unused.
        Dense(classes, activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10,
              batch_size=32, verbose=2)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_DIR / "sign_mnist_mlp.keras"
    model.save(output_path)
    print(f"Test accuracy: {accuracy:.4f} | test loss: {loss:.4f}")
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
