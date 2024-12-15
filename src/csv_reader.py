import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

class CSVReader:
    def __init__(self, file_dict: Dict[str, str]):
        self.file_dict = file_dict
        self.data_frames = {}
        self._load_all_files()
    
    def _load_file(self, name: str, url: str) -> None:
        try:
            df = pd.read_csv(url)
            self.data_frames[name] = df
        except Exception as e:
            print(f"Error loading {name}: {str(e)}")
            self.data_frames[name] = None
    
    def _load_all_files(self) -> None:
        """Load all CSV files using multiple threads"""
        with ThreadPoolExecutor(max_workers=min(len(self.file_dict), 5)) as executor:
            futures = {
                executor.submit(self._load_file, name, url): name 
                for name, url in self.file_dict.items()
            }
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                    print(f"Successfully loaded {name}")
                except Exception as e:
                    print(f"Failed to load {name}: {str(e)}")
    
    def get_data(self, key: str) -> pd.DataFrame:
        if key not in self.data_frames:
            raise KeyError(f"No data found for key: {key}")
        if self.data_frames[key] is None:
            raise ValueError(f"Data for {key} failed to load")
        return self.data_frames[key]
    
    def get_available_keys(self) -> list:
        return list(self.file_dict.keys())