from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time
import os
from functools import wraps
from skimage.metrics import structural_similarity as ssim_2d
from scipy.ndimage import gaussian_filter

import numpy as np


def timing(f):
    """Decorator for measuring the execution time of methods."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        elapsed = te - ts
        print("%r took %f s\n" % (f.__name__, elapsed))
        sys.stdout.flush()
        wrapper.elapsed_time = elapsed  # Guarda el tiempo en un atributo
        return result
    wrapper.elapsed_time = None  # Inicializa el atributo
    return wrapper

def merge_values(values):
    return np.hstack(values) if isinstance(values, (list, tuple)) else values


def trim_to_65535(x):
    """Incorrect output when batch size > 65535.
    https://github.com/tensorflow/tensorflow/issues/9870
    https://github.com/tensorflow/tensorflow/issues/13869
    """
    N = 65535
    if isinstance(x, (list, tuple)):
        return (x[0][:N], x[1][:N]), (x[0][N:], x[1][N:])
    return x[:N], x[N:]

def trim_to_65535_cartesian(x):
    N = 3920000 // 10 # Asuming  70x70x800 points
    if isinstance(x, (list, tuple)):
        return (x[0][:], x[1][:N]), (x[0][:], x[1][N:])
    return x[:N], x[N:]
  
def trim_to_65535_cartesian_vel(x):
    N = 1 # Asuming  70x70x800 points
    if isinstance(x, (list, tuple)):
        return (x[0][:N], x[1][:]), (x[0][N:], x[1][:])
    return x[:N], x[N:]

def sorting_key(name):
    import re
    """Función para ordenar por tamaño y luego por capas"""
    size = int(re.search(r'size_(\d+)_layers', name).group(1))
    layers = int(re.search(r'layers_(\d+)', name).group(1))
    return (size, layers)

def mean_squared_error_outlier(y_true, y_pred):
    error = np.ravel((y_true - y_pred) ** 2)
    error = np.sort(error)[: -len(error) // 1000]
    return np.mean(error)

def mean_squared_error(y_true, y_pred):
    error = np.ravel((y_true - y_pred) ** 2)
    return np.mean(error)
def ssim_3d(volume1, volume2, dynamic_range=None, win_size=7, gaussian_weights=True):
    """
    Calcula el SSIM 3D entre dos volúmenes (voxeles).
    
    Args:
        volume1, volume2 (np.ndarray): Volúmenes 3D a comparar.
        dynamic_range (float): Rango dinámico de los datos (max - min).
        win_size (int): Tamaño de la ventana para comparación local.
        gaussian_weights (bool): Si True, usa pesos gaussianos en la ventana.
    
    Returns:
        float: Valor SSIM promedio (similaridad estructural).
    """
    if volume1.shape != volume2.shape:
        raise ValueError("Los volúmenes deben tener las mismas dimensiones.")
    
    if dynamic_range is None:
        dynamic_range = volume1.max() - volume1.min()
    
    ssim_values = []
    for t in range(volume1.shape[0]):
        slice1 = volume1[t, :, :]
        slice2 = volume2[t, :, :]
        ssim_val = ssim_2d(
            slice1, slice2, 
            data_range=dynamic_range,
            win_size=win_size,
            gaussian_weights=gaussian_weights,
            multichannel=False
        )
        ssim_values.append(ssim_val)
    
    return np.mean(ssim_values)

def ssim_4d(volumes1, volumes2, dynamic_range=None, win_size=7, gaussian_weights=True):
    """
    Calcula el SSIM promedio para múltiples volúmenes 3D (4D array).

    Args:
        volumes1, volumes2 (np.ndarray): Arrays 4D de forma (n_volumes, z, y, x)
        dynamic_range (float): Rango dinámico de los datos
        win_size (int): Tamaño de la ventana
        gaussian_weights (bool): Usar pesos gaussianos

    Returns:
        float: SSIM promedio sobre todos los volúmenes
    """
    if volumes1.shape != volumes2.shape:
        raise ValueError("Los arrays deben tener las mismas dimensiones.")

    if len(volumes1.shape) != 4:
        raise ValueError("Se esperaban arrays 4D")

    if dynamic_range is None:
        dynamic_range = volumes1.max() - volumes1.min()

    ssim_values = []
    n_volumes = volumes1.shape[0]

    for i in range(n_volumes):
        # Calcular SSIM 3D para cada par de volúmenes
        ssim_val = ssim_3d(
            volumes1[i],
            volumes2[i],
            dynamic_range=dynamic_range,
            win_size=win_size,
            gaussian_weights=gaussian_weights,
        )
        ssim_values.append(ssim_val)

    return np.mean(ssim_values)

@timing
def safe_predict(model, data, X_test, y_test, fname=None):
    y_pred = model.predict(X_test)
    error = np.mean((y_test - y_pred) ** 2)
    print("Test MSE: {}".format(error))
    error_out = mean_squared_error_outlier(y_test, y_pred)
    print("Test MSE w/o outliers: {}\n".format(error_out))

    if fname is not None:
        np.savetxt(fname, np.hstack((X_test[1], y_test, y_pred)))
    return error


def eig(kernel, num, Nx, eigenfunction=True):
    """Compute the eigenvalues and eigenfunctions of a kernel on [0, 1].
    """
    h = 1 / (Nx - 1)
    c = kernel(np.linspace(0, 1, num=Nx)[:, None])[0] * h
    A = np.empty((Nx, Nx))
    for i in range(Nx):
        A[i, i:] = c[: Nx - i]
        A[i, i::-1] = c[: i + 1]
    A[:, 0] *= 0.5
    A[:, -1] *= 0.5

    if not eigenfunction:
        return np.flipud(np.sort(np.real(np.linalg.eigvals(A))))[:num]

    eigval, eigvec = np.linalg.eig(A)
    eigval, eigvec = np.real(eigval), np.real(eigvec)
    idx = np.flipud(np.argsort(eigval))[:num]
    eigval, eigvec = eigval[idx], eigvec[:, idx]
    for i in range(num):
        eigvec[:, i] /= np.trapz(eigvec[:, i] ** 2, dx=h) ** 0.5
    return eigval, eigvec


def trapz(y, dx):
    """Integrate [y(x1), y(x2), ...] or [[y1(x1), y1(x2), ...], [y2(x1), y2(x2), ...], ...]
    using the composite trapezoidal rule.

    Return: [I1(x1)=0, I1(x2), ...] or [[I1(x1)=0, I1(x2), ...], [I2(x1)=0, I2(x2), ...], ...]
    """
    if len(y.shape) == 1:
        left = np.cumsum(y)[:-1]
        right = np.cumsum(y[1:])
        return np.hstack(([0], (left + right) / 2 * dx))
    left = np.cumsum(y, axis=1)[:, :-1]
    right = np.cumsum(y[:, 1:], axis=1)
    return np.hstack((np.zeros((len(y), 1)), (left + right) / 2 * dx))


def make_triple(sensor_value, x, y, num):
    """For a `sensor_value` of `u`, a list of locations `x` and the corresponding solution `y`,
    generate a dataset of `num` triples.

    sensor_value: 1d array
    x: 2d array, N x d
    y: 1d array
    """
    idx = np.random.choice(len(x), size=num, replace=False)
    x = x[idx]
    y = y[idx][:, None]
    return np.hstack([np.tile(sensor_value, (num, 1)), x, y])

def quick_save(filename, x, y, path=os.getcwd()):
    # Revisar si existe el path
    file_path = os.path.join(path, filename)
    if not os.path.exists(path):
        print(f"Creando directorio: {path}")
        os.makedirs(path)

    # Revisar si en el archivo ya existe el x
    #if os.path.exists(file_path):
    #    with open(file_path,'r') as f:
    #          lines = f.readlines()
    #          for line in lines:
    #              if str(x)+"," in line:
    #                  print(f"El dato {x} ya existe en el archivo {file_path}.")
    
    # Agregar dato
    print(f"Agregando dato {x} al archivo {file_path}.")
    with open(file_path, 'a') as f:
        f.write(f"{x}, {y}\n")

def clean_checkpoints(models_path):
  """Remove all checkpoints in the model path except the best one (name is in the checkpoint file).""" 
  checkpoint_file = os.path.join(models_path, "checkpoint")
  with open(checkpoint_file, 'r') as f:
        for line in f:
            if line.startswith('model_checkpoint_path:'):
                best_model_file = os.path.basename(line.split('"')[1])
  print(f"Best model file to keep: {best_model_file}")
  for file in os.listdir(models_path):
    if file.startswith("model.ckpt") and not file.startswith(best_model_file):
        os.remove(os.path.join(models_path, file))
        print(f"Removed checkpoint: {file}")
        
def save_config(path,activation, initializer, layer_size, num_layers, dropout, optimizer, lr):
  import json
  if not os.path.exists(path):
        print(f"Creando directorio: {path}")
        os.makedirs(path)
  config = {
    "activation": activation,
    "initializer": initializer,
    "layer_sizes": layer_size,
    "num_layers": num_layers,
    "dropout": dropout,
    "optimizer": optimizer,
    "lr": lr
  }
  
  with open(os.path.join(path,"config.json"), 'w') as archivo:
    json.dump(config, archivo, indent=2)