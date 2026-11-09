import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.constants import DATA_NAME

output_path = Path(
    f"data/vel_models/{DATA_NAME}/model_59.npy"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

n_models = 7

velocities = np.linspace(1500, 4500, n_models).round().astype(int)
print(velocities)
models = np.array(velocities, dtype=np.float32).reshape(n_models, 1, 1, 1) * np.ones((n_models, 1, 100, 100), dtype=np.float32)

print(models.shape)           
print(models[0, 0].shape)
print(models[:, 0].min(), models[:, 0].max()) 

np.save(output_path, models)
