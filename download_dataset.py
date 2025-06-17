import os
import kaggle
from pathlib import Path

def download_plant_disease_dataset():
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Download the PlantVillage dataset
    # This is the dataset ID for PlantVillage dataset
    dataset = 'abdallahalidev/plantvillage-dataset'
    
    print("Downloading PlantVillage dataset...")
    kaggle.api.dataset_download_files(
        dataset,
        path=data_dir,
        unzip=True
    )
    print("Dataset downloaded successfully!")

if __name__ == "__main__":
    download_plant_disease_dataset() 