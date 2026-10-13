import numpy as np
from pathlib import Path

output_path = Path(
    "data/vel_models/homogenous/model_00.npy"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

velocities = np.array(
    [1500, 2000, 2500, 3000, 3500, 4000, 4500],
)

models = np.array(velocities, dtype=np.float32).reshape(7, 1, 1, 1) * np.ones((7, 1, 70, 70), dtype=np.float32)

print(models.shape)            # (7, 1, 70, 70)
print(models[0, 0].shape)      # (70, 70)
print(models[0, 0].min(), models[0, 0].max()) 

np.save(output_path, models)
