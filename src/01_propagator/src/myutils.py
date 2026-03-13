import os
import numpy as np
import time
from functools import wraps
import sys

# SpeedWave Samples
flat = sample = 1500 * np.ones((70, 70))

two_layers = np.ones((70, 70))  # crea una matriz de unos
two_layers[:35, :] = 2500  # mitad superior (filas 0 a 34) a 2500
two_layers[35:, :] = 4000  # mitad inferior (filas 35 a 69) a 4000


def find_files(path: str, file_type: str) -> list:
    """
    Searches for files or folders in a given directory, filtering by type.

    Parameters:
    ----------
    path : str
        The directory path where the search will be performed. Use "./" for the current path.

    file_type : str
        The type of element to search for:
        - "/" to search for folders only.
        - ".ext" (e.g., ".txt", ".py", etc.) to search for files with that extension.

    Returns:
    -------
    list
        A list of names (not full paths) of files or folders that match the specified criteria.

    Raises:
    ------
    FileNotFoundError
        If the provided path does not exist.

    Examples:
    --------
    >>> find_files("my/folder/", ".txt")
    ['file1.txt', 'notes.txt']

    >>> find_files("my/folder/", "/")
    ['subfolder1', 'images']
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path does not exist: {path}")

    elements = os.listdir(path)
    result = []

    for elem in elements:
        full_path = os.path.join(path, elem)

        if file_type == "/" and os.path.isdir(full_path):
            result.append(elem)
        elif (
            file_type.startswith(".")
            and os.path.isfile(full_path)
            and elem.endswith(file_type)
        ):
            result.append(elem)

    return result


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


def expand_edges_speedwave(speedwave, n_aumentos, FreeSurfaceLimit):
    """
    Expands the edges of a speedwave by replicating the first and last columns and rows.

    Parameters:
    speedwave (np.ndarray): The original speedwave data.
    n_aumentos (int): The number of times to replicate the edges.

    Returns:
    np.ndarray: The expanded speedwave.
    """
    expanded_wave = np.hstack(
        (
            speedwave[:, 0, np.newaxis].repeat(n_aumentos, axis=-1),
            speedwave,
            speedwave[:, -1, np.newaxis].repeat(n_aumentos, axis=-1),
        )
    )
    expanded_wave = np.vstack(
        (
            expanded_wave[0, np.newaxis].repeat(FreeSurfaceLimit, axis=0),
            # np.full((FreeSurfaceLimit, expanded_wave.shape[1]), 0.0),
            expanded_wave,
            expanded_wave[-1, np.newaxis].repeat(n_aumentos, axis=0),
        )
    )
    return expanded_wave
