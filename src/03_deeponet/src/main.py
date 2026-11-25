import os
import sys

import numpy as np

os.environ["DDE_BACKEND"] = "pytorch"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import deepxde as dde

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.utils import (
    clean_checkpoints,
    mean_squared_error_outlier,
    safe_predict,
    save_config,
    timing,
)
from src.constants import EXPERIMENT, SAMPLER, DATA_NAME
from src.myclasses import FullCartesianProd, MIONetFullCartesianProd


@timing
def train(model, iterations, checker, batch_size=None):
    return model.train(
        iterations=iterations,
        callbacks=[checker],
        batch_size=batch_size,
    )


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def load_training_data(data_path, n_train, n_points, n_locs):
    with np.load(os.path.join(data_path, "train.npz")) as data:
        y_train = data["y_train"][:n_locs, :n_train, :n_points].astype(np.float32)
        n_locs, n_train, n_points = y_train.shape
        X_train = (
            data["X_train2"][:n_locs].astype(np.float32),
            data["X_train0"][:n_train].astype(np.float32),
            data["X_train1"][:n_points].astype(np.float32),
        )

    with np.load(os.path.join(data_path, "val.npz")) as data:
        X_val = (
            data["X_val2"].astype(np.float32),
            data["X_val0"].astype(np.float32),
            data["X_val1"][:n_points].astype(np.float32),
        )
        y_val = data["y_val"][:n_locs, :, :n_points].astype(np.float32)

    return X_train, y_train, X_val, y_val, n_train, n_points, n_locs


def run(
    m,
    net,
    optimizer,
    lr,
    batch_size,
    iterations,
    data_path="data/",
    model_path="model/",
    result_path="results/",
    n_train=None,
    n_points=None,
    n_locs=None,
    loss_history_path="results/",
):
    X_train, y_train, X_val, y_val, n_train, n_points, n_locs = (
        load_training_data(data_path, n_train, n_points, n_locs)
    )

    print(
        f"X_train2 shape: {X_train[0].shape}, "
        f"X_train0 shape: {X_train[1].shape}, "
        f"X_train1 shape: {X_train[2].shape}, "
        f"y_train shape: {y_train.shape}"
    )
    print(
        f"X_val2 shape: {X_val[0].shape}, "
        f"X_val0 shape: {X_val[1].shape}, "
        f"X_val1 shape: {X_val[2].shape}, "
        f"y_val shape: {y_val.shape}"
    )

    data = FullCartesianProd(
        X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val
    )

    ensure_directory(model_path)
    ensure_directory(result_path)
    ensure_directory(loss_history_path)

    model = dde.Model(data, net)
    model.compile(optimizer, lr=lr, loss="MSE", metrics=[mean_squared_error_outlier])
    ckpt_path = os.path.join(model_path, "model.ckpt")
    checker = dde.callbacks.ModelCheckpoint(
        ckpt_path, save_better_only=False, period=1000, monitor="test loss",
    )
    losshistory, train_state = train(
        model,
        iterations,
        checker,
        batch_size=(n_locs, n_train, batch_size),
    )
    dde.utils.save_loss_history(
        losshistory,
        os.path.join(loss_history_path, "losshistory"),
    )

    ckpt_path = os.path.join(
        model_path,
        f"model.ckpt-{train_state.best_step}.ckpt",
    )
    model.restore(ckpt_path, verbose=1)
    clean_checkpoints(model_path)
    mse = safe_predict(model, data, X_val, y_val)

    np.savez(
        os.path.join(result_path, "train_state.npz"),
        best_loss_train=train_state.best_loss_train,
        best_loss_val=train_state.best_loss_test,
        best_step=train_state.best_step,
        elapsed_time=train.elapsed_time,
        mse_val=mse,
    )
    return train_state, train.elapsed_time



def resolve_data_dimensions(data_path, n_train, n_points, n_locs):
    with np.load(os.path.join(data_path, "train.npz")) as data:
        available_locs, available_train, available_points = data["y_train"].shape

    return (
        available_train if n_train is None else n_train,
        available_points if n_points is None else n_points,
        available_locs if n_locs is None else n_locs,
    )


def write_hyperparameters(
    path,
    parameter,
    experiment,
    sampler,
    m,
    iterations,
    dim_x,
    lr,
    batch_size,
    activation,
    initializer,
    optimizer,
    n_train,
    n_points,
    dropout,
    trunk_net,
    branch_net1,
    branch_net2,
    layer_size,
    num_layers,
):
    ensure_directory(path)
    hyperparameters = {
        "experiment": experiment,
        "parameter": parameter,
        "sampler": sampler,
        "m": m,
        "iterations": iterations,
        "dim_x": dim_x,
        "lr": lr,
        "batch_size": batch_size,
        "activation": activation,
        "initializer": initializer,
        "optimizer": optimizer,
        "n_train": n_train,
        "n_points": n_points,
        "dropout": dropout,
        "Trunk net": str(trunk_net),
        "Branch net1": str(branch_net1),
        "Branch net2": str(branch_net2),
        "Iterated net": "None",
        "layer_sizes": str(layer_size),
        "num_layers_list": str(num_layers),
    }

    with open(os.path.join(path, "hyperparameters.txt"), "w") as file:
        for key, value in hyperparameters.items():
            file.write(f"{key}: {value}\n")


def main():
    N_TRAIN = None
    N_POINTS = None
    N_LOCS = 10
    m = 100*100
    iterations = 2_000
    dim_loc = 2
    dim_x = 3
    optimizer = "adam"
    lr = 0.0001
    batch_size = 4096
    activation = "relu"
    initializer = "Glorot normal" 
    dropout = 0
    layer_size = 800
    num_layers = 10
    branch_net1 = [dim_loc] + [layer_size//2] * (num_layers//2) + [layer_size]# branch1 is smaller than branch2
    branch_net2 = [m] + [layer_size] * num_layers + [layer_size]
    trunk_net = [dim_x] + [layer_size] * num_layers
    net = MIONetFullCartesianProd(
        branch_net1,
        branch_net2,
        trunk_net,
        activation,
        initializer
    )
    PARAMETER = "alone" #f"alone_size{layer_size}_layers_{num_layers}"
    SOURCE_DATA_PATH =os.path.join("data")
    OUTPUT_DATA_PATH =os.path.join("output")

    data_path=os.path.join(SOURCE_DATA_PATH, "processed",DATA_NAME,"train_val",SAMPLER)#f"data/train_val/{SAMPLER}/"
    
    model_path=os.path.join(OUTPUT_DATA_PATH,"models",EXPERIMENT,PARAMETER)#f"model/{EXPERIMENT}/{PARAMETER}/"
    result_path=os.path.join(OUTPUT_DATA_PATH, "experiments",EXPERIMENT,PARAMETER)#f"experiments/{EXPERIMENT}/{PARAMETER}/"
    loss_history_path=result_path
    
    save_config(model_path,activation, initializer, layer_size, num_layers, dropout, optimizer, lr)

    N_TRAIN, N_POINTS, N_LOCS = resolve_data_dimensions(
        data_path,
        N_TRAIN,
        N_POINTS,
        N_LOCS,
    )
    write_hyperparameters(
        model_path,
        PARAMETER,
        EXPERIMENT,
        SAMPLER,
        m,
        iterations,
        dim_x,
        lr,
        batch_size,
        activation,
        initializer,
        optimizer,
        N_TRAIN,
        N_POINTS,
        dropout,
        trunk_net,
        branch_net1,
        branch_net2,
        layer_size,
        num_layers,
    )

    run(
        m,
        net,
        optimizer,
        lr,
        batch_size,
        iterations,
        n_points=N_POINTS,
        n_train=N_TRAIN,
        n_locs=N_LOCS,
        data_path=data_path,
        model_path=model_path,
        result_path=result_path,
        loss_history_path=loss_history_path,
    )


if __name__ == "__main__":
    main()
