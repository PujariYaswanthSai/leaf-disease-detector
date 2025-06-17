import os
import json
from pathlib import Path

def setup_kaggle_credentials(username, key):
    # Create .kaggle directory in user's home directory
    kaggle_dir = Path.home() / '.kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    # Create kaggle.json with credentials
    credentials = {
        "username": username,
        "key": key
    }
    
    kaggle_json = kaggle_dir / 'kaggle.json'
    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f)
    
    # Set appropriate permissions (600)
    os.chmod(kaggle_json, 0o600)
    
    print("Kaggle credentials have been set up successfully!")
    print(f"Credentials file location: {kaggle_json}")

if __name__ == "__main__":
    # Your Kaggle credentials
    username = "kushal8832445"
    key = "8ca8c7c159ded555ee2216109a7a9234"
    
    setup_kaggle_credentials(username, key) 