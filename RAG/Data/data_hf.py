import os
import pandas as pd
from datasets import load_dataset
from pathlib import Path
import json

class HuggingFaceDataDownloader:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.csv_dir = self.base_dir / "CSV"
        self.csv_dir.mkdir(exist_ok=True)
        
        self.datasets = {
            "argilla/farming": {
                "name": "argilla_farming",
                "description": "General farming dataset from Argilla"
            },
            "YuvrajSingh9886/Agriculture-Soil-QA-Pairs-Dataset": {
                "name": "agriculture_soil_qa",
                "description": "Agriculture Soil Question-Answer pairs dataset"
            },
            "KisanVaani/agriculture-qa-english-only": {
                "name": "kisanvaani_agriculture_qa",
                "description": "KisanVaani Agriculture QA in English only"
            },
            "YuvrajSingh9886/Agriculture-Irrigation-QA-Pairs-Dataset": {
                "name": "agriculture_irrigation_qa",
                "description": "Agriculture Irrigation Question-Answer pairs dataset"
            }
        }
        
        self.columns_to_remove = [
            'id', 'Id', 'ID', 
            'Unnamed: 0', 'unnamed: 0',
            'index', 'Index',
            '__index_level_0__'
        ]
    
    def clean_columns(self, df):
        original_columns = df.columns.tolist()        
        columns_to_drop = []
        for col in df.columns:
            if col in self.columns_to_remove:
                columns_to_drop.append(col)
            elif col.startswith('Unnamed:') or col.startswith('unnamed:'):
                columns_to_drop.append(col)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"    🗑️  Removed columns: {columns_to_drop}")
        
        return df
    
    def clean_data_for_csv(self, df):
        df = self.clean_columns(df)        
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].apply(lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x) if x is not None else '')
        return df
    
    def download_dataset(self, dataset_id, dataset_info):
        print(f"Downloading dataset: {dataset_id}")
        
        try:
            dataset = load_dataset(dataset_id)            
            for split_name, split_data in dataset.items():
                print(f"  Processing split: {split_name} ({len(split_data)} examples)")
                df = pd.DataFrame(split_data)
                df_clean = self.clean_data_for_csv(df.copy())
                if len(dataset.items()) > 1:
                    filename = f"{dataset_info['name']}_{split_name}.csv"
                else:
                    filename = f"{dataset_info['name']}.csv"
                
                filepath = self.csv_dir / filename
                df_clean.to_csv(filepath, index=False, encoding='utf-8')
                print(f"    Saved: {filepath} ({len(df_clean)} rows, {len(df_clean.columns)} columns)")
            
            return True
            
        except Exception as e:
            print(f"    Error downloading {dataset_id}: {str(e)}")
            return False
    
    def download_all_datasets(self):
        print("Starting download of agricultural datasets from Hugging Face...")
        print(f"Saving to directory: {self.csv_dir}")
        print("=" * 70)
        
        successful_downloads = 0
        total_datasets = len(self.datasets)
        
        for dataset_id, dataset_info in self.datasets.items():
            print(f"\n[{successful_downloads + 1}/{total_datasets}] {dataset_info['description']}")
            if self.download_dataset(dataset_id, dataset_info):
                successful_downloads += 1
            print("-" * 50)
        
        print(f"Successful: {successful_downloads}/{total_datasets}")
        print(f"Failed: {total_datasets - successful_downloads}/{total_datasets}")
        print(f"Files saved in: {self.csv_dir}")
        
        csv_files = list(self.csv_dir.glob("*.csv"))
        if csv_files:
            print(f"\nDownloaded files:")
            for file in csv_files:
                file_size = file.stat().st_size / 1024  # KB
                print(f"  • {file.name} ({file_size:.1f} KB)")

def main():
    print("Agricultural Dataset Downloader")
    print("Downloading datasets from Hugging Face Hub...")
    
    downloader = HuggingFaceDataDownloader()
    downloader.download_all_datasets()
    
    print("\nDownload process completed!")

if __name__ == "__main__":
    try:
        import datasets
        import pandas as pd
    except ImportError as e:
        print(f"Missing required packages. Please install:")
        print("pip install datasets pandas")
        exit(1)
    
    main()
