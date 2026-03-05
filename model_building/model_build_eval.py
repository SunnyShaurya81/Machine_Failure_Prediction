from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os
import joblib
import numpy as np

# ===============================================
# Define project directories within the script
# ===============================================
project_dir = "."
data_dir = os.path.join(project_dir, "data")
model_dir = os.path.join(project_dir, "model_building")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# --- Configuration for Hugging Face Repositories ---
project_dir_relative = "."
data_dir_relative = os.path.join(project_dir_relative, "data")
model_dir_relative = os.path.join(project_dir_relative, "model_building")
dataset_repo_id = 'SunnyShaurya1981/engine-predictive-maintenance-data3'
dataset_repo_type = 'dataset'
model_repo_id = "SunnyShaurya1981/engine-predictive-maintenance-model"
model_repo_type = "model"

# --- Initialize HF API ---
api = HfApi(token=os.getenv("HF_TOKEN"))

# --- Load Train and Test Data from Hugging Face ---
print("Loading train.csv from Hugging Face...")
train_dataset = load_dataset(dataset_repo_id, data_files={'train': 'train.csv'}, split='train')
train_df = train_dataset.to_pandas()
print("train.csv loaded successfully!")

print("Loading test.csv from Hugging Face...")
test_dataset = load_dataset(dataset_repo_id, data_files={'test': 'test.csv'}, split='test')
test_df = test_dataset.to_pandas()
print("test.csv loaded successfully!")

# Separate features and target
X_train = train_df.drop("Engine Condition", axis=1)
y_train = train_df["Engine Condition"]

X_test = test_df.drop("Engine Condition", axis=1)
y_test = test_df["Engine Condition"]

# --- Model Building and Tuning (Random Forest) ---
print("Starting Random Forest model tuning...")
rf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
print(f"Best Parameters for Random Forest: {grid.best_params_}")

# --- Evaluate Tuned Model ---
print("Evaluating tuned Random Forest model on test set...")
rf_preds = best_rf.predict(X_test)

accuracy = accuracy_score(y_test, rf_preds)
precision = precision_score(y_test, rf_preds)
recall = recall_score(y_test, rf_preds)
f1 = f1_score(y_test, rf_preds)

print(f"Tuned Random Forest Accuracy: {accuracy:.4f}")
print(f"Tuned Random Forest Precision: {precision:.4f}")
print(f"Tuned Random Forest Recall: {recall:.4f}")
print(f"Tuned Random Forest F1 Score: {f1:.4f}")

# --- Save the best model locally ---
print("Saving the best Random Forest model locally...")
model_save_path = os.path.join(model_dir, 'random_forest_model.joblib')
joblib.dump(best_rf, model_save_path)
print(f"Model saved successfully locally to: {model_save_path}")

# --- Register Best Model to Hugging Face Model Hub ---

# Check if the model space exists and create it if it doesn't
try:
    api.repo_info(repo_id=model_repo_id, repo_type=model_repo_type)
    print(f"Model space '{model_repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model space '{model_repo_id}' not found. Creating new space...")
    try:
        create_repo(repo_id=model_repo_id, repo_type=model_repo_type, private=False)
        print(f"Model space '{model_repo_id}' created.")
    except HfHubHTTPError as e:
        print(f"Error creating repository: {e}")
        print("Please ensure your HF_TOKEN is valid and has write access.")
        exit(1)
except HfHubHTTPError as e:
    print(f"Error checking repository: {e}")
    print("Please ensure your HF_TOKEN is valid and has read access.")
    exit(1)

# Upload the locally saved model to the Hugging Face model repository
print(f"Uploading model to Hugging Face Model Hub: {model_repo_id}...")
try:
    api.upload_file(
        path_or_fileobj=model_save_path,
        path_in_repo="random_forest_model.joblib",
        repo_id=model_repo_id,
        repo_type=model_repo_type,
        commit_message="Upload best Random Forest model"
    )
    print("Random Forest model uploaded successfully to Hugging Face Model Hub!")
except Exception as e:
    print(f"Error uploading model to Hugging Face: {e}")
    print("Please check your model_repo_id, HF_TOKEN permissions, and the file path.")
    exit(1)

print("Model registration process completed.")
