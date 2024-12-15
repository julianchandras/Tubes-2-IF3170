import pickle
import os
from datetime import datetime

class ModelPersistence:
    @staticmethod
    def save_model(model, filepath, metadata=None):
        model_data = {
            'model': model,
            'params': model.get_params() if hasattr(model, 'get_params') else None,
            'type': model.__class__.__name__,
            'timestamp': datetime.now(),
            'metadata': metadata
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        return model_data

