from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os

# --- Configuration ---
# Your Hugging Face Space repository ID (e.g., "your-username/your-space-name")
# This will be the name of your deployed Streamlit app
SPACE_REPO_ID = "SunnyShaurya1981/engine-predictive-maintenance-app" # Replace with your desired Space ID
SPACE_REPO_TYPE = "space"

# Local directory where the deployment files are located
# This path aligns with where app.py, requirements.txt, Dockerfile were created
LOCAL_DEPLOY_DIR = "." # Now that files are copied, '.' is correct relative to script execution in model_building

# List of files to upload
DEPLOY_FILES = [
    "app.py",
    "requirements.txt",
    "Dockerfile",
]

# --- Initialize HF API ---
# Assumes HF_TOKEN is set as an environment variable (e.g., from Colab secrets or environment)
api = HfApi(token=os.getenv("HF_TOKEN"))

# --- Create Hugging Face Space if it doesn't exist ---
try:
    api.repo_info(repo_id=SPACE_REPO_ID, repo_type=SPACE_REPO_TYPE)
    print(f"Hugging Face Space '{SPACE_REPO_ID}' already exists. Proceeding with upload.")
except RepositoryNotFoundError:
    print(f"Hugging Face Space '{SPACE_REPO_ID}' not found. Creating new Space...")
    try:
        # Corrected space_sdk to 'docker' since a Dockerfile is provided
        create_repo(repo_id=SPACE_REPO_ID, repo_type=SPACE_REPO_TYPE, private=False, space_sdk='docker')
        print(f"Hugging Face Space '{SPACE_REPO_ID}' created successfully.")
    except HfHubHTTPError as e:
        print(f"Error creating Hugging Face Space: {e}")
        print("Please ensure your HF_TOKEN is valid and has write access for Space creation.")
        exit(1)
except HfHubHTTPError as e:
    print(f"Error checking Hugging Face Space: {e}")
    print("Please ensure your HF_TOKEN is valid and has read access.")
    exit(1)

# --- Upload Deployment Files ---
print(f"Uploading deployment files to Hugging Face Space '{SPACE_REPO_ID}'...")
for filename in DEPLOY_FILES:
    local_file_path = os.path.join(LOCAL_DEPLOY_DIR, filename)
    if not os.path.exists(local_file_path):
        print(f"Error: File '{local_file_path}' not found. Skipping.")
        continue

    try:
        api.upload_file(
            path_or_fileobj=local_file_path,
            path_in_repo=filename, # Upload to the root of the Space
            repo_id=SPACE_REPO_ID,
            repo_type=SPACE_REPO_TYPE,
            commit_message=f"Add {filename} for Streamlit app deployment"
        )
        print(f"Successfully uploaded {filename}.")
    except Exception as e:
        print(f"Error uploading {filename}: {e}")
        print("Please check your SPACE_REPO_ID, HF_TOKEN permissions, and the file path.")

print("Deployment script finished.")
