import os
import numpy as np
import multiprocessing
from constants import EXPERIMENT, SAMPLER, DATA_NAME
from utils import save_config

def worker_iterate_n_train(n_train, gpu_id, m, epochs, dim_x, optimizer,lr, activation, initializer, n_points):
    # Configura la GPU visible ANTES de importar tensorflow
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import tensorflow as tf
    from tensorflow.keras import backend as K
    import deepxde as dde
    import main

    # Limpia sesión anterior
    K.clear_session()

    # (Opcional) limita memoria si usas GPUs compartidas
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    print(f"\033[32m Ejecutando n_train = {n_train} en GPU {gpu_id} \033[0m")

    # Crear modelo
    net = dde.maps.DeepONetCartesianProd(
        [m, 50, 50],
        [dim_x, 50, 50],
        activation,
        initializer,
    )

    # Crear paths personalizados
    result_dir = f"output/experiments/experiment7/{n_train}/"
    model_dir = f"model/experiment7/{n_train}/"
    os.makedirs(result_dir, exist_ok=True)

    # Ejecutar entrenamiento
    train_state, elapsed_time = main.run(
        m, net, optimizer, lr, epochs,
        data_path="data/",
        model_path=model_dir,
        result_path=result_dir,
        n_train=n_train,
        n_points=n_points
    )   

    print(f"\033[34m Finalizado n_train = {n_train}. Tiempo: {elapsed_time:.2f}s, Loss train: {train_state.best_loss_train:.3e}, loss test: {train_state.best_loss_test} at step {train_state.best_step} \033[0m")

def worker_iterate_network_size(layer_size, num_layers, gpu_id, m, epochs, dim_x, optimizer, lr, dropout, batch_size, activation, initializer, n_train, n_points):
    # Configura la GPU visible ANTES de importar tensorflow
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import tensorflow as tf
    from tensorflow.keras import backend as K
    import deepxde as dde
    import main

    # Limpia sesión anterior
    K.clear_session()

    # (Opcional) limita memoria si usas GPUs compartidas
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    print(f"\033[32m Ejecutando capas={num_layers}, tamaño={layer_size} en GPU {gpu_id} \033[0m")

    # Crear modelo con la configuración de tamaño y profundidad
    branch_net = [m] + [layer_size] * num_layers
    trunk_net = [dim_x] + [layer_size] * num_layers
    
    net = dde.maps.DeepONetCartesianProd(
        branch_net,
        trunk_net,
        activation,
        initializer,
        dropout_rate=dropout,
    )

    # Crear paths personalizados
    result_dir = f"output/experiments/{EXPERIMENT}/size_{layer_size}_layers_{num_layers}/"
    model_dir = f"model/{EXPERIMENT}/size_{layer_size}_layers_{num_layers}/"
    data_path=f"data/train_val/{SAMPLER}/"
    os.makedirs(result_dir, exist_ok=True)

    # Ejecutar entrenamiento
    train_state, elapsed_time = main.run(
        m, net, optimizer, lr, batch_size, epochs,
        data_path=data_path,
        model_path=model_dir,
        result_path=result_dir,
        loss_history_path=result_dir,
        n_train=n_train,
        n_points=n_points
    )   
    
    save_config(model_dir,activation, initializer, layer_size, num_layers, dropout, optimizer, lr)

    print(f"\033[34m Finalizado capas={num_layers}, tamaño={layer_size}. Tiempo: {elapsed_time:.2f}s, Loss train: {train_state.best_loss_train:.3e}, loss test: {train_state.best_loss_test} at step {train_state.best_step} \033[0m")

def worker_iterate_data_size(n_train, n_points, gpu_id, m, epochs, dim_x, optimizer, lr, dropout, batch_size, activation, initializer, layer_size, num_layers):
    # Configura la GPU visible ANTES de importar tensorflow
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import tensorflow as tf
    from tensorflow.keras import backend as K
    import deepxde as dde
    import main
    import numpy as np

    # Limpia sesión anterior
    K.clear_session()

    # (Opcional) limita memoria si usas GPUs compartidas
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    print(f"\033[32m Ejecutando n_train={n_train}, n_points={n_points} en GPU {gpu_id} \033[0m")

    branch_net = [m] + [layer_size] * num_layers
    trunk_net = [dim_x] + [layer_size] * num_layers
    # Crear modelo
    net = dde.maps.DeepONetCartesianProd(
        branch_net,
        trunk_net,
        activation,
        initializer,
        dropout_rate=dropout,
    )

    # Crear paths personalizados
    result_dir = f"output/experiments/{EXPERIMENT}/data_ntrain_{n_train}_npoints_{n_points}/"
    model_dir = f"model/{EXPERIMENT}/data_ntrain_{n_train}_npoints_{n_points}/"
    data_path=f"data/train_val/{SAMPLER}/"
    os.makedirs(result_dir, exist_ok=True)

    # Ejecutar entrenamiento
    train_state, elapsed_time = main.run(
        m, net, optimizer, lr, batch_size, epochs,
        data_path=data_path,
        model_path=model_dir,
        result_path=result_dir,
        loss_history_path=result_dir,
        n_train=n_train,
        n_points=n_points
    )   
    
    save_config(model_dir,activation, initializer, layer_size, num_layers, dropout, optimizer, lr)

    print(f"\033[34m Finalizado n_train_frac={n_train}, n_points_frac={n_points}. Tiempo: {elapsed_time:.2f}s, Loss train: {train_state.best_loss_train:.3e}, loss test: {train_state.best_loss_test} at step {train_state.best_step} \033[0m")

def iterate_network_size():
    multiprocessing.set_start_method("spawn", force=True)

    m = 4900
    epochs = 100_000
    dim_x = 3
    optimizer = "adam"
    lr = 0.0001
    dropout = 0
    batch_size = 4096
    activation = "relu"
    initializer = "Glorot normal"
    n_points = None
    n_train = None  # Valor fijo para este experimento

    layer_sizes = [60, 330, 600]
    num_layers_list = [2, 6, 10]
    gpu_indices = [5, 6, 7]  # Lista de índices de las GPUs disponibles,
    
    if n_train is None:
        data_path=f"data/train_val/{SAMPLER}/"
        with np.load(data_path + "train.npz") as d:
          n_train, _ = d["y_train"].shape
    if n_points is None:
        data_path=f"data/train_val/{SAMPLER}/"
        with np.load(data_path + "train.npz") as d:
          _, n_points = d["y_train"].shape

    # Crear todas las combinaciones posibles
    combinations = [(size, layers) for size in layer_sizes for layers in num_layers_list]
    # Guardar en texto plano los hiperparámetros
    if not os.path.exists(f"output/experiments/{EXPERIMENT}"):
        os.makedirs(f"output/experiments/{EXPERIMENT}")
    with open(f"output/experiments/{EXPERIMENT}/hyperparameters.txt", "w") as f:
        f.write(f"experiment: {EXPERIMENT}\n")
        f.write(f"sampler: {SAMPLER}\n")
        f.write(f"m: {m}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"dim_x: {dim_x}\n")
        f.write(f"optimizer: {optimizer}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"activation: {activation}\n")
        f.write(f"initializer: {initializer}\n")
        f.write(f"n_train: {n_train}\n")
        f.write(f"n_points: {n_points}\n")
        f.write(f"dropout: {dropout}\n")
        f.write("Iterated net: Both" + "\n")
        f.write("layer_sizes_list: " + ", ".join(map(str, layer_sizes)) + "\n")
        f.write("num_layers_list: " + ", ".join(map(str, num_layers_list)) + "\n")
    processes = []
    for i, (layer_size, num_layers) in enumerate(combinations):
        gpu_id = gpu_indices[i % len(gpu_indices)]
        p = multiprocessing.Process(target=worker_iterate_network_size, args=(
            layer_size, num_layers, gpu_id, m, epochs, dim_x, optimizer, lr, dropout, batch_size, activation, 
            initializer, n_train, n_points
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def iterate_data_size():
    multiprocessing.set_start_method("spawn", force=True)

    m = 4900
    epochs = 100_000
    dim_x = 3
    optimizer = "adam"
    lr = 0.0001
    batch_size = 4096
    activation = "relu"
    initializer = "Glorot normal"
    dropout = 0
    
    layer_size = 600
    num_layers = 10
    

    n_train_sizes = [0.1, 0.55, 1]
    n_points_sizes = [0.1, 0.55, 1]
    
    # Cargar datos para determinar tamaños
    data_path=f"data/processed/{DATA_NAME}/train_val/{SAMPLER}/"
    with np.load(data_path + "train.npz") as d:
        total_n_train, _ = d["y_train"].shape
        _, total_n_points = d["y_train"].shape

    n_train_sizes = [int(total_n_train*f) for f in n_train_sizes] # fraccionar el total en la lista
    n_points_sizes = [int(total_n_points*f) for f in n_points_sizes] # fraccionar el total en la lista
    gpu_indices = [6, 7]  # Lista de índices de las GPUs disponibles

    combinations = [(n_train, n_points) for n_train in n_train_sizes for n_points in n_points_sizes]
    # Guardar en texto plano los hiperparámetros
    if not os.path.exists(f"output/experiments/{EXPERIMENT}"):
        os.makedirs(f"output/experiments/{EXPERIMENT}")
    with open(f"output/experiments/{EXPERIMENT}/hyperparameters.txt", "w") as f:
        f.write(f"experiment: {EXPERIMENT}\n")
        f.write(f"sampler: {SAMPLER}\n")
        f.write(f"m: {m}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"dim_x: {dim_x}\n")
        f.write(f"optimizer: {optimizer}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"activation: {activation}\n")
        f.write(f"initializer: {initializer}\n")
        f.write(f"layer_sizes: {layer_size}\n")
        f.write(f"num_layers: {num_layers}\n")
        f.write(f"dropout: {dropout}\n")
        f.write("n_train_list: " + ", ".join(map(str, n_train_sizes)) + "\n")
        f.write("n_points_list: " + ", ".join(map(str, n_points_sizes)) + "\n")
    processes = []
    for i, (n_train_frac, n_points_frac) in enumerate(combinations):
        gpu_id = gpu_indices[i % len(gpu_indices)]
        p = multiprocessing.Process(target=worker_iterate_data_size, args=(
            n_train_frac, n_points_frac, gpu_id, m, epochs, dim_x, optimizer, lr, dropout, batch_size, activation, initializer, layer_size, num_layers
        ))
        p.start()
        processes.append(p)
    

if __name__ == "__main__":
    #iterate_network_size()
    iterate_data_size()
    print("Todos los experimentos han finalizado.")