import os
import sys
import numpy as np

sys.path.append(os.path.dirname( os.path.dirname(__file__)))
from utils import change_resolution_speedwave


INPUT_FOLDER = os.path.join("data","vel_models","open_fwi")
OUTPUT_FOLDER = os.path.join("data","vel_models","open_fwi_100")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".npy"):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        # Load the original array
        data = np.load(input_path)
        data_resized = np.empty((data.shape[0], data.shape[1], 100, 100), dtype=data.dtype)
        for model in range(data.shape[0]):
            model_data = data[model, 0, :, :]
            resized_data = change_resolution_speedwave(model_data, (100, 100))
            data_resized[model, 0, :, :] = resized_data


        # Save the cropped array to the new folder        
        np.save(output_path, data_resized)
        print(f"Processed and saved: {output_path}")
        # show shape
        print(f"Original shape: {data.shape}, Resized shape: {data_resized.shape}")