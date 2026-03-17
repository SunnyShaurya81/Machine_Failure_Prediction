
# ---------------Import Required Libraries--------------
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import os

# --------------Define the Hugging Face Space-----------
SPACE_REPO_ID = "SunnyShaurya1981/engine-predictive-maintenance-app"
SPACE_REPO_TYPE = "space"

# --------------Correct deployment directory------------
#LOCAL_DEPLOY_DIR = "../deployment"
LOCAL_DEPLOY_DIR = "deployment"

# --------------Files to Upload------------------------
DEPLOY_FILES = [
    "app.py",
    "requirements.txt",
    "Dockerfile",
]

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create Hugging Face Space if not exists
try:
    api.repo_info(repo_id=SPACE_REPO_ID, repo_type=SPACE_REPO_TYPE)
    print(f"Space '{SPACE_REPO_ID}' already exists.")
except RepositoryNotFoundError:
    print("Creating Hugging Face Space...")
    create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="docker",
        private=False
    )

# Upload files
print("Uploading deployment files...")

for filename in DEPLOY_FILES:

    local_file_path = os.path.join(LOCAL_DEPLOY_DIR, filename)

    if not os.path.exists(local_file_path):
        print(f"{local_file_path} not found")
        continue

    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=filename,
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        commit_message=f"Upload {filename}"
    )

    print(f"{filename} uploaded successfully")

print("Deployment finished.")
