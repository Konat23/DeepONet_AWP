import numpy as np
from pathlib import Path

output_path = Path(
    "data/vel_models/homogenous/model_00.npy"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

n_models = 500

velocities = np.linspace(1500, 4500, n_models).round().astype(int)

models = np.array(velocities, dtype=np.float32).reshape(n_models, 1, 1, 1) * np.ones((n_models, 1, 70, 70), dtype=np.float32)

print(models.shape)            # (7, 1, 70, 70)
print(models[0, 0].shape)      # (70, 70)
print(models[0, 0].min(), models[0, 0].max()) 

np.save(output_path, models)
