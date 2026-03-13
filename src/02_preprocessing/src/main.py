import os
import numpy as np

from sklearn.preprocessing import MaxAbsScaler
import joblib

import wavefield_samplers
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.constants import DATA_NAME
from src.utils import timing
# ========== PATH CONFIGURATION ==========
print(f"actual path: {os.path.abspath(__file__)}")
SOURCE_DATA_PATH =os.path.join("data") 
WAVEFIELD_PATH = os.path.join(SOURCE_DATA_PATH,"wavefields",DATA_NAME)
VELMODEL_PATH = os.path.join(SOURCE_DATA_PATH,"vel_models",DATA_NAME)

output_path = os.path.join(SOURCE_DATA_PATH, "processed",DATA_NAME,"train_val")
scaler_path = os.path.join(SOURCE_DATA_PATH, "processed",DATA_NAME,"train_val")
reference_path = os.path.join(SOURCE_DATA_PATH, "processed",DATA_NAME,"ref")

# ================ SEED  ================
seed = 42
np.random.seed(seed)

@timing
def load_wavefield_data(path, wave_num):
    print("\n=== Loading Wavefield Data ===")
    # Determinar cuantos paquetes de 500 hay que cargar segun wave_num
    num_files = (wave_num-1) // 500 + 1
    for i in range(num_files):
        # Cargar el paquete de 500
        file_path = os.path.join(path, f"wavefield_{i:02d}.npy")
        if i == 0:
            data = np.load(file_path)
        else:
            data = np.concatenate((data, np.load(file_path)), axis=0)
        print(f"Loaded file: {file_path}, current shape: {data.shape}")
    # Carga solo los datos necesarios
    data = data[:wave_num, :, :, :]  # Select the first wave_num
    print("Wavefield data loaded successfully.")
    print(f"Wavefield data shape: {data.shape}")
    print(f"Wavefield data type: {data.dtype}")
    return np.array(data)

@timing
def load_velmodel_data(path, wave_num):
    print("\n=== Loading Velocity model Data ===")
    # Determinar cuantos paquetes de 500 hay que cargar segun wave_num
    num_files = (wave_num-1) // 500 + 1
    for i in range(num_files):
        # Cargar el paquete de 500
        file_path = os.path.join(path, f"model_{i:02d}.npy")
        if i == 0:
            data = np.load(file_path, mmap_mode=None)
        else:
            data = np.concatenate((data, np.load(file_path, mmap_mode=None)), axis=0)
    # Carga solo los datos necesarios 
    data = data[:wave_num, :, :]  # Select the first wave_num
    print("Velocity model data loaded successfully.")
    print(f"Velocity model data shape: {data.shape}")
    return data

def load_wavefield_data_ref(path, wave_num, npack=59):
    print("\n=== Loading Wavefield Reference Data ===")
    file_path = os.path.join(path, f"wavefield_{npack:02d}.npy")
    data = np.load(file_path, mmap_mode='r')
    data = data[:wave_num, :, :, :]  # Select the first wave_num
    print("Wavefield reference data loaded successfully.")
    print(f"Wavefield reference data shape: {data.shape}")
    return np.array(data)

def load_velmodel_data_ref(path, wave_num, npack=59):
    print("\n=== Loading Velocity model Reference Data ===")
    file_path = os.path.join(path, f"model_{npack:02d}.npy")
    data = np.load(file_path, mmap_mode='r')
    data = data[:wave_num, :, :]  # Select the first wave_num
    print("Velocity model reference data loaded successfully.")
    print(f"Velocity model reference data shape: {data.shape}")
    return data

def flat_velmodel_data(velmodel_data):
    print("\n=== Flattening Velocity model Data ===")
    # Assuming Velocity model_data is a 3D array with shape (samples, height, width)
    # Flattening the last two dimensions
    velmodel_data = velmodel_data.reshape(velmodel_data.shape[0], -1)
    print("Velocity model data flattened successfully.")
    print(f"Flattened Velocity model data shape: {velmodel_data.shape}")
    return velmodel_data

def velmodel_repeat(velmodel_data, npoints_output):
    print("\n=== Repeating Velocity Model Data ===")
    # Repeat the Velocity Model data to match the number of points in the wavefield data

    velmodel_data = np.repeat(velmodel_data, npoints_output, axis=0)
    print("Velocity Model data repeated successfully.")
    print(f"Repeated Velocity Model data shape: {velmodel_data.shape}")
    return velmodel_data

def split_cartesian_data(velmodel_data, wave_loc, wave_value, split_percent):
    print("\n=== Splitting Cartesian Data ===")
    # Get the number of samples
    num_samples = velmodel_data.shape[0]
    # Calculate the split index
    split_index = int(num_samples * split_percent)
    indices = np.arange(num_samples)
    # Split the data into training and validation sets
    velmodel_train = velmodel_data[indices[:split_index]]
    velmodel_val = velmodel_data[indices[split_index:]]
    wave_loc_train = wave_loc[indices[:split_index]]
    wave_loc_val = wave_loc[indices[split_index:]]
    wavefield_train = wave_value[indices[:split_index]]
    wavefield_val = wave_value[indices[split_index:]]
    
    print("Data split successfully.")
    print(f"Training set size: {velmodel_train.shape[0]}")
    print(f"Validating set size: {velmodel_val.shape[0]}")
    
    return (velmodel_train, wave_loc_train, wavefield_train), (velmodel_val, wave_loc_val, wavefield_val)

def mix_samples_data(velmodel_data, wavefield_data):
    print("\n=== Mixing Samples Data ===")
    if not os.path.exists(os.path.join(output_path)):
      os.makedirs(os.path.join(output_path))
    # Randomly mix the Velocity model data and wavefield data.
    samples = velmodel_data.shape[0]
    index = np.arange(samples)
    np.random.shuffle(index)
    np.save(os.path.join(output_path,"shuffle_index.npy"), index)
    velmodel_data = velmodel_data[index]
    wavefield_data = wavefield_data[index]
    print("Data mixed successfully.")
    print(f"Mixed Velocity model data shape: {velmodel_data.shape}")
    print(f"Mixed wavefield data shape: {wavefield_data.shape}")
    return velmodel_data, wavefield_data

def mix_cartesian_data(velmodel_data, wavefield_data):
    print("\n=== Mixing Cartesian Data ===")
    samples = velmodel_data.shape[0]
    index = np.arange(samples)
    np.random.shuffle(index)
    velmodel_data = velmodel_data[index]
    wavefield_data = wavefield_data[index]
    print("Data mixed successfully.")
    print(f"Mixed Velocity model data shape: {velmodel_data.shape}")
    print(f"Mixed wavefield data shape: {wavefield_data.shape}")
    return velmodel_data, wavefield_data

def split_data(velmodel_data, wavefield_data, split_percent):
    print("\n=== Splitting Data ===")
    # Get the number of samples
    num_samples = velmodel_data.shape[0]
    # Calculate the split index
    split_index = int(num_samples * split_percent)
    
    # Split the data into training and validating sets
    velmodel_train = velmodel_data[:split_index]
    velmodel_val = velmodel_data[split_index:]
    
    wavefield_train = wavefield_data[:split_index]
    wavefield_val = wavefield_data[split_index:]
    
    print("Data split successfully.")
    print(f"Training set size: {velmodel_train.shape[0]}")
    print(f"Validating set size: {velmodel_val.shape[0]}")
    
    return (velmodel_train, wavefield_train), (velmodel_val, wavefield_val)

@timing
def normalize_data(velmodel_train, wavefield_train, velmodel_val, wavefield_val, normalize_by="by_all"):
    print("\n=== Normalizing Data ===")
    # Normalize the velmodel and wavefield data.
  # Normalize the velmodel and wavefield data.
    if normalize_by == "by_all":
        print("Normalizing by all train data...")
        scaler_velmodel = MaxAbsScaler()
        scaler_wavefield = MaxAbsScaler()
        
        velmodel_train = scaler_velmodel.fit_transform(velmodel_train.reshape(-1, 1)).reshape(velmodel_train.shape)
        velmodel_val = scaler_velmodel.transform(velmodel_val.reshape(-1, 1)).reshape(velmodel_val.shape)
        
        wavefield_train = scaler_wavefield.fit_transform(wavefield_train.reshape(-1, 1)).reshape(wavefield_train.shape)
        wavefield_val = scaler_wavefield.transform(wavefield_val.reshape(-1, 1)).reshape(wavefield_val.shape)
        
        # Save the scalers
        os.makedirs(scaler_path, exist_ok=True)
        joblib.dump(scaler_velmodel, os.path.join(scaler_path, "scaler_velmodel.pkl"))
        joblib.dump(scaler_wavefield, os.path.join(scaler_path, "scaler_wavefield.pkl"))
        
    elif normalize_by == "by_sample":
        # Normalize each sample individually. We dont need to save the scalers
        print("Normalizing by sample...")
        scaler_velmodel = MaxAbsScaler()
        scaler_wavefield = MaxAbsScaler()
        
        velmodel_train = scaler_velmodel.fit_transform(velmodel_train.T).T
        velmodel_val = scaler_velmodel.fit_transform(velmodel_val.T).T
        
        # To-do: Normalizar wavefield_train y wavefield_val por sample
        wavefield_train = scaler_wavefield.fit_transform(wavefield_train.reshape(-1, 1)).reshape(wavefield_train.shape)
        wavefield_val = scaler_wavefield.transform(wavefield_val.reshape(-1, 1)).reshape(wavefield_val.shape)
          
    else:
        raise ValueError(f"{normalize_by} is not a valid normalization method")
    
    print("Data normalized successfully.")
    
    return velmodel_train, wavefield_train, velmodel_val, wavefield_val

def normalize_cartesian_data(velmodel_train,wave_loc_train, wavefield_train, velmodel_val, wave_loc_val, wavefield_val, sample_type, normalize_by="by_all"):
    print("\n=== Normalizing Cartesian Data ===")
    # Crear scalers con tipos reducidos
    scaler_velmodel = MaxAbsScaler()
    scaler_wave_loc = MaxAbsScaler()
    scaler_wavefield = MaxAbsScaler()
    
    # Normalizar por lotes si es necesario
    velmodel_train = scaler_velmodel.fit_transform(velmodel_train.reshape(-1, 1)).reshape(velmodel_train.shape)
    velmodel_val = scaler_velmodel.transform(velmodel_val.reshape(-1, 1)).reshape(velmodel_val.shape)
    
    wave_loc_train = scaler_wave_loc.fit_transform(wave_loc_train)
    wave_loc_val = scaler_wave_loc.transform(wave_loc_val)
    
    if normalize_by == "by_all":
      wavefield_train = scaler_wavefield.fit_transform(wavefield_train.reshape(-1, 1)).reshape(wavefield_train.shape)
      wavefield_val = scaler_wavefield.transform(wavefield_val.reshape(-1, 1)).reshape(wavefield_val.shape)
      # Save the scalers
      os.makedirs(os.path.join(scaler_path,sample_type,"scalers"), exist_ok=True)
      joblib.dump(scaler_velmodel, os.path.join(scaler_path,sample_type,"scalers", "scaler_velmodel.pkl"))
      joblib.dump(scaler_wave_loc, os.path.join(scaler_path,sample_type,"scalers", "scaler_wave_loc.pkl"))
      joblib.dump(scaler_wavefield, os.path.join(scaler_path,sample_type,"scalers", "scaler_wavefield.pkl"))
    elif normalize_by == "by_sample":
      wavefield_train = scaler_wavefield.fit_transform(wavefield_train.T).T
      wavefield_val = scaler_wavefield.fit_transform(wavefield_val.T).T

    print("Data normalized successfully.")
    return velmodel_train, wave_loc_train, wavefield_train, velmodel_val, wave_loc_val, wavefield_val

def sample_cartesian_data(wavefield_data, sample_type, n_snaps=36):
    # n_snaps: Si no se usa por snaps, se calcula su equivalente
    print(f"\n=== Sampling Cartesian Data: {sample_type} ===")
    _, nt, nz, nx = wavefield_data.shape
    if sample_type == "random":
        npoints_output = nz*nx*n_snaps # Equivalente a n_snaps snapshots de 70x70
        wave_loc, wave_value = wavefield_samplers.random(wavefield_data, npoints_output)
    elif sample_type == "snapshot":
      n_snapshots = n_snaps # Number of snapshots to use. n_snaps snapshots of 70x70 (20%)
      snapshots = np.linspace(0, wavefield_data.shape[1]-1, num=n_snapshots, endpoint=True).astype(int)
      npoints_output = len(snapshots)*nz*nx
      wave_loc, wave_value = wavefield_samplers.select_snapshots(wavefield_data, snapshots)
    elif sample_type == "sobol":
      npoints_output = n_snaps*nz*nx # Equivalente a n_snaps snapshots 
      wave_loc, wave_value = wavefield_samplers.sobol(wavefield_data, npoints_output)
    elif sample_type == "random_cone":
      npoints_output = n_snaps*nz*nx # Equivalente a n_snaps snapshots de 70x70 (20%)
      wave_loc, wave_value = wavefield_samplers.random_cone(wavefield_data, npoints_output)
    elif sample_type == "sobol_cone":
      npoints_output = n_snaps*nz*nx
      wave_loc, wave_value = wavefield_samplers.sobol_cone(wavefield_data, npoints_output)
    else:
      raise ValueError(f"{sample_type} is not valid sample_type")
    del wavefield_data # free memory
    print(f"Sampling completed. Number of points: {npoints_output}")
    return wave_loc, wave_value

def save_data(velmodel, wave_loc, wavefield_value, dataset_type, sample_type):
    print(f"\n=== Saving {dataset_type.capitalize()} Data ===")
    if not os.path.exists(os.path.join(output_path, sample_type)):
      os.makedirs(os.path.join(output_path, sample_type))
    print("Saving data...")
    print(f"{dataset_type} shapes")
    print(f"(X{dataset_type}0), (X{dataset_type}1), (y{dataset_type})")
    print(f"{velmodel.shape}, {wave_loc.shape}, {wavefield_value.shape}")
    print(f"Saving train data...")
    kwargs = {
          f"X_{dataset_type}0": velmodel,
          f"X_{dataset_type}1": wave_loc,
          f"y_{dataset_type}": wavefield_value
      }
    np.savez(os.path.join(output_path, sample_type, f"{dataset_type}.npz"), **kwargs)
    print(f"{dataset_type.capitalize()} data saved successfully.")

def save_reference_data(wavefield, velmodel, ref=None, path="results"):
    print("\n=== Saving Reference Data ===")
    # if path does not exist, create it
    if not os.path.exists(path):
      os.makedirs(path)
    print("Saving wavield reference data... shape", wavefield[ref].shape)
    np.save(os.path.join(path,"yref_test.npy"), wavefield[ref])
    print(f"Saving velmodel reference data...shape {velmodel[ref,0].shape}")
    np.save(os.path.join(path,"xref_test.npy"), velmodel[ref,0])
    print("Reference data saved successfully.")

def main(sample_type, n_snaps, split_percent = 0.8, wave_num = 100, n_ref = 1, normalize_by="by_all"):
    print("\n=== Starting Data Processing ===")
    """
    Main function to process propagation data for wavefield and velmodel.

    Args:
      sample_type (str): Specifies the type of sampling to be performed. 
      Options are:
      - "fullsnapshot": Use a fixed number of snapshots from the wavefield data.
      - "random": Randomly sample points from the wavefield data.
      - "randomsnapshot": Placeholder for future implementation.
      - "sobol": Use Sobol sampling for the wavefield data.
      - "random_cone": Randomly sample points within a cone-shaped region.
      - "sobol_cone": Use Sobol sampling within a cone-shaped region.
      Default is "random".
      split_percent (float): The percentage of data to use for training. Default is 0.8.
      wave_num (int): The number of wavefield and velmodel samples to load. Must be greater than 0.
      n_ref (int): The number of reference samples to save. Default is 1.
      normalize_by (str): Specifies the normalization method. Options are:
      - "by_all": Normalize using all data.
      - "by_sample": Normalize each sample individually. Default is "byall".

    Raises:
      ValueError: If an invalid `sample_type` is provided.

    """
  
    ######################################
    assert wave_num > 0, "wave_num must be greater than 0"
    
    if n_ref > 0:
      ref_idx = np.arange(0, n_ref)
      wavefield_data_ref = load_wavefield_data_ref(WAVEFIELD_PATH, n_ref, npack=59)
      velmodel_data_ref = load_velmodel_data_ref(VELMODEL_PATH, n_ref, npack=59)
      save_reference_data(wavefield_data_ref, velmodel_data_ref ,ref=ref_idx, path=reference_path)
    else:
      print("No reference data to save.")    
    
    velmodel_data = load_velmodel_data(VELMODEL_PATH, wave_num)
    wavefield_data = load_wavefield_data(WAVEFIELD_PATH, wave_num)
    
    velmodel_data = flat_velmodel_data(velmodel_data)
    
    velmodel_data, wavefield_data = mix_samples_data(velmodel_data, wavefield_data)
    
    (velmodel_train, wavefield_train), (velmodel_val, wavefield_val) = split_data(velmodel_data, 
                                                                                      wavefield_data, 
                                                                                      split_percent)
    del (velmodel_data, wavefield_data) # free memory
    
    velmodel_train, wavefield_train, velmodel_val, wavefield_val = normalize_data(velmodel_train, wavefield_train, 
                                                                                      velmodel_val, wavefield_val, normalize_by=normalize_by)
    wave_loc_train, wave_value_train  = sample_cartesian_data(wavefield_train, sample_type,n_snaps=n_snaps)
    wave_loc_val, wave_value_val  = sample_cartesian_data(wavefield_val, sample_type, n_snaps=n_snaps)
    
    save_data(velmodel_train, wave_loc_train, wave_value_train, dataset_type="train", sample_type=sample_type)
    save_data(velmodel_val, wave_loc_val, wave_value_val, dataset_type="val", sample_type=sample_type)
    print("\n=== Data Processing Completed Successfully ===")
    
if __name__ == "__main__":
    main(sample_type="sobol",n_snaps=36, wave_num=10, n_ref=10, normalize_by="by_all")
