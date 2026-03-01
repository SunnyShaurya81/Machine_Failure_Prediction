from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os

HF_TOKEN = os.getenv("HF_TOKEN").strip()
SPACE_ID = "SunnyShaurya1981/engine-predictive-maintenance-app"

api = HfApi(token=HF_TOKEN)

# Define a dictionary mapping the path in the repo to the local file path relative to deploy.py
# deploy.py is in model_building/
# app.py is in ../app.py
# requirements.txt is in ./requirements.txt
# Dockerfile is in ./Dockerfile
DEPLOY_FILES_MAP = {
    "app.py": "../app.py",
    "requirements.txt": "./requirements.txt",
    "Dockerfile": "./Dockerfile",
}

# --- Create Hugging Face Space if it doesn't exist ---
try:
    api.repo_info(repo_id=SPACE_ID, repo_type="space")
    print(f"Hugging Face Space '{SPACE_ID}' already exists. Proceeding with upload.")
except RepositoryNotFoundError:
    print(f"Hugging Face Space '{SPACE_ID}' not found. Creating new Space...")
    try:
        create_repo(repo_id=SPACE_ID, repo_type="space", private=False, space_sdk='docker')
        print(f"Hugging Face Space '{SPACE_ID}' created successfully.")
    except HfHubHTTPError as e:
        print(f"Error creating Hugging Face Space: {e}")
        print("Please ensure your HF_TOKEN is valid and has write access for Space creation.")
        exit(1)
except HfHubHTTPError as e:
    print(f"Error checking Hugging Face Space: {e}")
    print("Please ensure your HF_TOKEN is valid and has read access.")
    exit(1)

# --- Upload Deployment Files ---
print(f"Uploading deployment files to Hugging Face Space '{SPACE_ID}'...")
for path_in_repo, local_relative_path in DEPLOY_FILES_MAP.items():
    if not os.path.exists(local_relative_path):
        print(f"Error: File '{local_relative_path}' not found. Please ensure it is correctly placed. Skipping.")
        continue

    try:
        api.upload_file(
            path_or_fileobj=local_relative_path,
            path_in_repo=path_in_repo,
            repo_id=SPACE_ID,
            repo_type="space",
            commit_message=f"Add {path_in_repo} for Streamlit app deployment"
        )
        print(f"Successfully uploaded {path_in_repo}.")
    except Exception as e:
        print(f"Error uploading {path_in_repo}: {e}")
        print("Please check your SPACE_ID, HF_TOKEN permissions, and the file path.")

print("Deployment script finished!")
