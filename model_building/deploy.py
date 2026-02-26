from huggingface_hub import HfApi
import os

HF_TOKEN = os.getenv("HF_TOKEN").strip() # Added .strip() to clean the token
SPACE_ID = "SunnyShaurya1981/engine-predictive-maintenance-app"

api = HfApi(token=HF_TOKEN)

files = ["app.py", "requirements.txt", "Dockerfile"]

for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message=f"Update {file}"
    )

print("Deployment successful!")
