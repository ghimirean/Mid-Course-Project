import pandas as pd
import joblib
from sklearn.metrics import classification_report

# 1. Load the datasets
print("Loading datasets...")
train_df = pd.read_csv('data/sign_mnist_train.csv')
test_df = pd.read_csv('data/sign_mnist_test.csv')

# 2. Separate features (X) and labels (y)
y_train = train_df['label']
X_train = train_df.drop('label', axis=1)

y_test = test_df['label']
X_test = test_df.drop('label', axis=1)

# 3. Normalize the pixel values (scale them from 0-255 to 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape} | y_test shape: {y_test.shape}")
print("Data preprocessing complete!")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 4. Train the Random Forest model
print("Training the Random Forest model (this might take a minute)...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Make predictions on the test set
print("Evaluating the model...")
y_pred = rf_model.predict(X_test)

# 6. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")



# 7. Generate a detailed performance report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save the trained model to disk
model_path = 'models/rf_baseline_model.pkl'
joblib.dump(rf_model, model_path)
print(f"\nModel saved successfully to {model_path}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 8. Generate and save the Confusion Matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save the plot
plt.savefig('results/confusion_matrix.png')
print("Confusion matrix saved successfully to results/confusion_matrix.png")