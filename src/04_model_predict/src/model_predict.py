"""
Predict a propagation using the DeepONet model
"""

import os
import time
import joblib
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.utils import mean_squared_error_outlier, trim_to_65535_cartesian,  quick_save, trim_to_65535_cartesian_vel, sorting_key
from src.constants import EXPERIMENT, SAMPLER, NORMALIZE_BY, DATA_NAME
import json
import glob
import tensorflow as tf


# ========== PATH CONFIGURATION ==========
INPUT_PATH = os.path.join("data")
OUTPUT_PATH = os.path.join("output")
#os.path.join("/mnt/compartida/sguerrero_cps/kioto_backup/deeponet_acustic_wave")
DATA_PATH = os.path.join(INPUT_PATH, "processed",DATA_NAME,"train_val", SAMPLER)
REF_PATH =  os.path.join(INPUT_PATH, "processed",DATA_NAME, "ref")
SCALER_DIR = os.path.join(INPUT_PATH,"processed",DATA_NAME, "train_val")
MODEL_PATH = os.path.join(OUTPUT_PATH,"models", EXPERIMENT)
RESULTS_PATH = os.path.join(OUTPUT_PATH, "experiments", EXPERIMENT)

# Number of Models to test
N_test = 500
# ========== TF CONFIGURATION ==========
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import deepxde as dde

# ========== DATA LOADING ==========
def get_data(path):
    train = np.load(os.path.join(path, "train.npz"))
    val = np.load(os.path.join(path,"val.npz"))

    X_train = (train["X_train0"], train["X_train1"])
    y_train = train["y_train"]

    x_val = (val["X_val0"], val["X_val1"])
    y_val = val["y_val"]

    print(f"[DATA] X_train shapes: {X_train[0].shape}, {X_train[1].shape}")
    print(f"[DATA] y_train shape: {y_train.shape}")
    print(f"[DATA] X_val shapes: {x_val[0].shape}, {x_val[1].shape}")
    print(f"[DATA] y_val shape: {y_val.shape}")

    data = dde.data.TripleCartesianProd(X_train=X_train, y_train=y_train, X_test=x_val, y_test=y_val)
    
    y_test = np.load(os.path.join(REF_PATH, "yref_test.npy"))[N_test:N_test+1] 
    x_test = np.load(os.path.join(REF_PATH, "xref_test.npy"))[N_test:N_test+1] 
    
    if y_test.ndim == 3:
        y_test = y_test.reshape(1,-1, 70, 70)
    if x_test.ndim == 2:
        x_test = x_test.reshape(1,70,70)

    print(f"[DATA] y_ref shape: {y_test.shape}")
    print(f"[DATA] x_ref shape: {x_test.shape}")
    

    return data, y_test, x_test

# ========== MODEL SETUP ==========
def get_model_name(checkpoint_file):
    with open(checkpoint_file, 'r') as f:
        for line in f:
            if line.startswith('model_checkpoint_path:'):
                return line.split('"')[1]
    return None

def get_model_architecture(problem, config):
    m = 70*70
    dim_x = 3
    layer_size = config["layer_sizes"]
    num_layers = config["num_layers"]
    branch_net = [m] + [layer_size] * num_layers
    trunk_net = [dim_x] + [layer_size] * num_layers
    dim_x = 3
    net = dde.maps.DeepONetCartesianProd(branch_net, trunk_net, config["activation"],  config["initializer"])
    return net

def get_model_config(path):
  config_path = os.path.join(path, "config.json")
  with open(config_path, "r") as f:
    config = json.load(f)
  return config

def safe_predict(model, x, batch_size_0=100, batch_size_1=3920000): #FIXME: Solo para prueba
    y_pred = []
    
    print(f"[RUN] Using manual batch sizes: x0={batch_size_0}, x1={batch_size_1}")
    
    start_time = time.time()
    total_combinations = 0
    
    for i in range(0, len(x[0]), batch_size_0):
        x0_batch = x[0][i:i + batch_size_0]
        
        for j in range(0, len(x[1]), batch_size_1):
            x1_batch = x[1][j:j + batch_size_1]
            
            X_add = (x0_batch, x1_batch)
            batch_pred = model.predict(X_add)
            y_pred.append(batch_pred)
            
            current_combinations = len(x0_batch) * len(x1_batch)
            total_combinations += current_combinations
            #print(f"[RUN] Batch: {len(x0_batch)}×{len(x1_batch)} = {current_combinations:,} comb. Total: {total_combinations:,}")
    
    total_time = time.time() - start_time
    print(f"[RUN] Total: {total_combinations:,} combinations in {total_time:.2f} seconds")
    return np.hstack(y_pred), total_time

def prepare_input_data(xref, is_cartesian, shapes=(1000,70,70)):
    speedwave = xref.reshape(-1, 1)
    t_range, x_range, z_range = np.arange(shapes[0]), np.arange(shapes[1]), np.arange(shapes[2])
    tt, xx, zz = np.meshgrid(t_range, x_range, z_range, indexing='ij')
    locations = np.column_stack((tt.ravel(), xx.ravel(), zz.ravel()))

    #scaler_wave_loc = joblib.load(os.path.join(SCALER_DIR, "scaler_wave_loc.pkl")) ya no se usa
    scaler_speed = joblib.load(os.path.join(SCALER_DIR, "scaler_velmodel.pkl"))
    locations = locations.astype(np.float32)#/ np.max(locations, axis=0)
    speedwave = scaler_speed.transform(speedwave).reshape(xref.shape[0],-1)
    
    total_points = locations.shape[0]
    if not is_cartesian:
        speedwave = np.tile(speedwave, (1, total_points)).T
    else:
      pass

    print(f"[INPUT] Speedwave shape: {speedwave.shape}")
    print(f"[INPUT] Locations shape: {locations.shape}")
    return (speedwave, locations)

def prepare_output_data(y_ref, is_cartesian):
  if NORMALIZE_BY == "by_sample":
    y_max = np.max(y_ref, axis=(1, 2, 3))
    y_ref = y_ref / y_max[:, None, None, None]
  elif NORMALIZE_BY == "by_all":
    scaler_wavefield = joblib.load(os.path.join(SCALER_DIR, "scaler_wavefield.pkl"))
    y_ref = scaler_wavefield.transform(y_ref.reshape(-1,1)).reshape(y_ref.shape)
  print(f"[OUTPUT] y_ref shape: {y_ref.shape}")
  return y_ref
    
# ========== MAIN PIPELINE ==========

def main():
    problem = "prop"
    save_prediction = True
    model_dirs = glob.glob(os.path.join(MODEL_PATH, "*"))
    print(MODEL_PATH)
    print(model_dirs)
    # Repetir primera inferencia para quitar tiempo de arranque    
    #model_dirs = sorted(model_dirs, key=sorting_key) #TODO: generalizar el ordenado
    #model_dirs = model_dirs[:1]+ model_dirs[:] # (repite la primera) FIXME: Quitar esta linea   
    print(f"[INFO] Experiment: {EXPERIMENT}, Sampler: {SAMPLER}, Normalize by: {NORMALIZE_BY}")
    print(f"[INFO] Found {len(model_dirs)} model directories.")
    
    print("[INFO] Loading data...")
    data, y_test, x_test = get_data(DATA_PATH)        
    x_test = prepare_input_data(x_test, is_cartesian=True, shapes=y_test[0,:,:,:].shape)
    y_test = prepare_output_data(y_test, is_cartesian=True)
    
    for n, model_dir in enumerate(model_dirs[:]):
        PARAMETER = os.path.basename(model_dir)
        MODEL_P_PATH = os.path.join(MODEL_PATH, PARAMETER)
        RESULTS_P_PATH = os.path.join(RESULTS_PATH, PARAMETER)

        if not os.path.exists(RESULTS_P_PATH):
          raise ValueError(f"[ERROR] Results path does not exist: {RESULTS_P_PATH}")
            #os.makedirs(RESULTS_P_PATH)

        # Clear TensorFlow session to reset the graph
        tf.keras.backend.clear_session()

        model_name = get_model_name(os.path.join(MODEL_P_PATH, "checkpoint"))
        print("="*100)
        print(f"[INFO] Problem type: {problem}")
        print(f"[INFO] Model checkpoint: {model_name}")
        print(f"[INFO] Paths:\n  Data: {DATA_PATH}\n  Model: {MODEL_P_PATH}\n  Results: {RESULTS_P_PATH}")

        print("[INFO] Building model architecture...")
        config = get_model_config(MODEL_P_PATH)
        net = get_model_architecture(problem, config)
        model = dde.Model(data, net) #TODO: Como compilar el modelo sin tener que cargar data(?)
        model.compile(config["optimizer"], lr=config["lr"],loss="MSE", metrics=[mean_squared_error_outlier]) #TODO: mse o parametrizar
        model.restore(os.path.join(MODEL_P_PATH, model_name), verbose=1)

        print("="*100)
        print("[INFO] Running prediction...")
        
        y_pred, total_time = safe_predict(model, x_test, batch_size_0=100, batch_size_1=39_200) #FIXME: Solo para prueba
        y_pred = y_pred.reshape(y_test.shape)
        print("[INFO] Prediction done")
        print(f"[TIMING] Total prediction time: {total_time:.2f} seconds")
        if n != 0: #TODO: Quitar condicion
          quick_save("timing.txt",PARAMETER, total_time, path=RESULTS_PATH)
        print(f"[RESULT] y_pred shape: {y_pred.shape}")
        print(f"[RESULT] y_ref shape: {y_test.shape}")
        
        #continue #TODO: Quitar esta linea, salta el guardado del resultado
        output_path = os.path.join(RESULTS_P_PATH, "ypred_test.npy") #
        
        if save_prediction: #and n != 0:  # Skip saving for the first run
            np.save(output_path, y_pred)
            print(f"Saved at {output_path}")

if __name__ == "__main__":
    main()
