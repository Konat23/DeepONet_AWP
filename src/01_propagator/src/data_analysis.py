import numpy as np
from  myplotters import plot_x_z_t_slices, plot_velmodel
from myutils import find_files
import os 

WAVEFIELD_PATH = os.path.join("data", "wavefields", "mydataset_ricker")
SPEEDWAVE_PATH = os.path.join("data", "speedwaves")

def load_wavefield_data(path, idx):
  data = None
  for i in idx:
    file_name = f"wavefield_{i:02d}.npy"
    file_path = os.path.join(path, file_name)
    if not os.path.exists(file_path):
      print(f"File {file_name} does not exist. Skipping.")
      continue
    print(f"Loading file {file_name}")
    if data is None:
      data = np.load(file_path, mmap_mode='r')
    else:
      data = np.concatenate((data, np.load(file_path, mmap_mode='r')), axis=0)
  if data is None:
    raise ValueError("No valid files were loaded.")
  print("Wavefield data loaded successfully.")
  print(f"Wavefield data shape: {data.shape}")
  print(f"  Samples: {data.shape[0]}")
  print(f"  Time steps: {data.shape[1]}")
  print(f"  X: {data.shape[2]}")
  print(f"  Z: {data.shape[3]}")
  return np.array(data)

def load_speedwave_data(path, idx):
    files = find_files(path, ".npy")
    for i, f in enumerate(files):
        if idx is not None and i not in idx:
            # Si el índice no está en la lista, saltar este archivo
            continue
        file_path = os.path.join(path, f)
        if i == 0:
            data = np.load(file_path)
        else:
            data = np.concatenate((data, np.load(file_path)), axis=0)
    print("Speedwave data loaded successfully.")
    print(f"Speedwave data shape:{data.shape}")
    print(f"  Samples: {data.shape[0]}")
    print(f"  X: {data.shape[2]}")
    print(f"  Z: {data.shape[3]}")
    return np.array(data)
  
def compute_basic_stats(data):
    print(f"Data type: {data.dtype}")
    print(f"Max value: {np.max(data)}")
    print(f"Min value: {np.min(data)}")
    #print(f"Mean value: {np.mean(data)}")
    #print(f"Standard Deviation: {np.std(data)}")
def layers(speedwave):
    """
    Compute the number of layers of each speedwave model.
    Then compute the frecuency of each layer number.
    """
    n_layers = []
    for i in range(speedwave.shape[0]):
        model = speedwave[i,0,:,:]
        layers = []
        for z in range(model.shape[0]):  # Iterate over depth (z-axis)
            current_layer = model[z, :]
            if len(layers) == 0 or not np.array_equal(current_layer, layers[-1]):
              layers.append(current_layer)
        n_layers.append(len(layers))
    n_layers = np.array(n_layers)
    unique, counts = np.unique(n_layers, return_counts=True)
    print("Number of layers frequency:")
    for u, c in zip(unique, counts):
        print(f"  {u} layers: {c} models")
    return n_layers
def compute_speed_stats(speedwave):
    n_layers = layers(speedwave)

print("\n============ SPEEDWAVE ============ ")
file_idx = [0,1,2,3,59] # Files to load (0,) (0,1,2,3) or list(range(10)), None to load all
speedwave =load_speedwave_data(SPEEDWAVE_PATH, file_idx)
compute_basic_stats(speedwave)
compute_speed_stats(speedwave)

file_idx = [0,1,2,3,59] # Files to load: (0,) (0,1,2,3) or list(range(10)), None to load all
print("\n============ WAVEFIELD ============ ")
wavefield = load_wavefield_data(WAVEFIELD_PATH, file_idx)
compute_basic_stats(wavefield)


