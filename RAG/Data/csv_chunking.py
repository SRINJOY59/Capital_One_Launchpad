import os
import pandas as pd
from pathlib import Path
import shutil

class CSVChunker:
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.csv_dir = self.base_dir / "CSV"
        self.csv_dir.mkdir(exist_ok=True)
        self.chunk_size = 100
    
    def process_csv(self, input_csv_path):
        print(f"Processing CSV: {input_csv_path}")
        
        try:
            df = pd.read_csv(input_csv_path)
            total_rows = len(df)
            
            print(f"   Total rows: {total_rows}")
            print(f"   Creating chunks of {self.chunk_size} rows each")
            
            chunk_number = 1
            created_files = []
            
            for start_row in range(0, total_rows, self.chunk_size):
                end_row = min(start_row + self.chunk_size, total_rows)
                
                print(f"   📝 Processing chunk {chunk_number}: rows {start_row + 1}-{end_row}")
                
                chunk_df = df.iloc[start_row:end_row].copy()
                
                input_name = Path(input_csv_path).stem
                chunk_filename = f"{input_name}_chunk_{chunk_number:02d}.csv"
                chunk_filepath = self.csv_dir / chunk_filename
                
                chunk_df.to_csv(chunk_filepath, index=False)
                
                created_files.append(chunk_filepath)
                print(f"   Saved: {chunk_filename}")
                
                chunk_number += 1
            
            original_path = Path(input_csv_path)
            if original_path.exists():
                original_path.unlink()
                print(f"    Removed original file: {original_path.name}")
            
            print(f"   Created {len(created_files)} chunks successfully!")
            return created_files
            
        except Exception as e:
            print(f"   ❌ Error processing CSV: {str(e)}")
            return []
    
    def process_all_csvs_in_directory(self, source_dir=None):
        if source_dir is None:
            source_dir = self.base_dir
        
        search_paths = [
            self.csv_dir
        ]
        
        csv_files = []
        
        for search_path in search_paths:
            if search_path.exists():
                found_csvs = list(search_path.glob("*.csv"))
                for csv in found_csvs:
                    if not '_chunk_' in csv.name and csv not in csv_files:
                        csv_files.append(csv)
                        
        print(f"Searched in the following directories:")
        for path in search_paths:
            exists = "✓" if path.exists() else "✗"
            csv_count = len(list(path.glob("*.csv"))) if path.exists() else 0
            print(f"   {exists} {path} ({csv_count} CSVs)")
        
        if not csv_files:
            print("\nNo CSV files found to process")
            print("Place CSV files in any of the searched directories above")
            return
        
        print(f"\nFound {len(csv_files)} CSV file(s) to process:")
        for csv_file in csv_files:
            print(f"   • {csv_file}")
        print("=" * 60)
        
        total_chunks = 0
        
        for csv_file in csv_files:
            chunks = self.process_csv(csv_file)
            total_chunks += len(chunks)
            print("-" * 60)
        
        print(f"\nProcessing Summary:")
        print(f"   Files processed: {len(csv_files)}")
        print(f"   Total chunks created: {total_chunks}")
        print(f"   Chunks saved in: {self.csv_dir}")
        
        chunk_files = list(self.csv_dir.glob("*.csv"))
        if chunk_files:
            print(f"\nCreated chunk files:")
            for chunk_file in sorted(chunk_files):
                file_size = chunk_file.stat().st_size / 1024
                print(f"   • {chunk_file.name} ({file_size:.1f} KB)")
    
    def get_csv_info(self, csv_path):
        try:
            df = pd.read_csv(csv_path, nrows=1)
            total_rows = len(pd.read_csv(csv_path))
            info = {
                'rows': total_rows,
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'file_size_mb': Path(csv_path).stat().st_size / 1024 / 1024
            }
            return info
        except Exception as e:
            return {'error': str(e)}

def main():
    print("CSV Chunker - Extract 500-row chunks from large CSV files")
    print("Splits large datasets while preserving column structure")
    print("=" * 70)
    
    chunker = CSVChunker()
    
    chunker.process_all_csvs_in_directory()
    
    print("\nCSV chunking process completed!")
    print(f"Check the '{chunker.csv_dir}' directory for processed chunks")

if __name__ == "__main__":
    main()
