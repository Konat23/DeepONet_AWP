from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import json
os.environ["DDE_BACKEND"] = "pytorch"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(5)
import deepxde as dde
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from src.utils import mean_squared_error, mean_squared_error_outlier, safe_predict, trim_to_65535, timing, clean_checkpoints, save_config
from src.constants import EXPERIMENT, SAMPLER, DATA_NAME

@timing
def train(model, epochs, checker, batch_size=None):
  return model.train(epochs=epochs, callbacks=[checker], batch_size=batch_size)


def run(m, net, optimizer, lr, batch_size, epochs, data_path="data/",model_path="model/",result_path="results/",n_train=None, n_points=None, loss_history_path="results/"):
    # Train    
    d = np.load(os.path.join(data_path,"train.npz"))
    X_train, y_train = (d["X_train0"][:n_train].astype(np.float32), d["X_train1"][:n_points].astype(np.float32)), d["y_train"][:n_train,:n_points].astype(np.float32)
    
    # Validation
    d = np.load(os.path.join(data_path,"val.npz"))
    X_val, y_val = (d["X_val0"].astype(np.float32), d["X_val1"][:n_points].astype(np.float32)), d["y_val"][:,:n_points].astype(np.float32)
    
    if n_train is None:
        n_train, _ = y_train.shape
    print(f"X_train0 shape: {X_train[0].shape},X_train1 shape: {X_train[1].shape}, y_train shape: {y_train.shape}")
    print(f"X_val0 shape: {X_val[0].shape},X_val1 shape: {X_val[1].shape}, y_val shape: {y_val.shape}")

    data = dde.data.TripleCartesianProd(
        X_train=X_train, y_train=y_train, X_test=X_val, y_test=y_val
    )
    # Crear  paths si no existe
    if not os.path.exists(model_path):
        print(f"Creando directorio: {model_path}")
        os.makedirs(model_path)
    if not os.path.exists(result_path):
        print(f"Creando directorio: {result_path}")
        os.makedirs(result_path)
    if not os.path.exists(loss_history_path):
        print(f"Creando directorio: {loss_history_path}")
        os.makedirs(loss_history_path)
        
    model = dde.Model(data, net)
    model.compile(optimizer, lr=lr, loss="MSE", metrics=[mean_squared_error_outlier])
    ckpt_path = os.path.join(model_path,"model.ckpt")
    checker = dde.callbacks.ModelCheckpoint(
        ckpt_path, save_better_only=False, period=1000, monitor="test loss",
    ) 
    losshistory, train_state = train(model, epochs, checker, batch_size=(n_train, batch_size)) 
    dde.utils.save_loss_history(losshistory, loss_history_path + "/losshistory")
    
    ckpt_path = os.path.join(model_path,"model.ckpt-" + str(train_state.best_step)+".ckpt")
    model.restore(ckpt_path, verbose=1)
    clean_checkpoints(model_path) #save_better_only 
    mse = safe_predict(model, data, X_val, y_val) 
    # Guardar train_state en numpy
    np.savez(os.path.join(result_path, "train_state.npz"),
             best_loss_train=train_state.best_loss_train,
             best_loss_val=train_state.best_loss_test,
             best_step=train_state.best_step,
             elapsed_time=train.elapsed_time,
             mse_val=mse)
    return train_state, train.elapsed_time



def main():    
    N_TRAIN = None
    N_POINTS = None
    m = 70*70
    epochs = 100_000
    dim_x = 3
    optimizer = "adam"
    lr = 0.0001
    batch_size = 4096
    activation = "relu"
    initializer = "Glorot normal" 
    dropout = 0
    layer_size = 800
    num_layers = 10
    branch_net = [m] + [layer_size] * num_layers
    trunk_net = [dim_x] + [layer_size] * num_layers
    net = dde.maps.DeepONetCartesianProd(
        branch_net,
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
    
    with open(os.path.join(model_path, "hyperparameters.txt"), "w") as f:
        f.write(f"experiment: {EXPERIMENT}\n")
        f.write(f"parameter: {PARAMETER}\n")
        f.write(f"sampler: {SAMPLER}\n")
        f.write(f"m: {m}\n")
        f.write(f"epochs: {epochs}\n")
        f.write(f"dim_x: {dim_x}\n")
        f.write(f"lr: {lr}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"activation: {activation}\n")
        f.write(f"initializer: {initializer}\n")
        f.write(f"optimizer: {optimizer}\n")
        f.write(f"n_train: {N_TRAIN}\n")
        f.write(f"n_points: {N_POINTS}\n")
        f.write(f"dropout: {dropout}\n")
        f.write("Trunk net: " + str(trunk_net) + "\n")
        f.write("Branch net: " + str(branch_net) + "\n")
        f.write("Iterated net: None" + "\n")
        f.write("layer_sizes: " + str(layer_size) + "\n")
        f.write("num_layers_list: " + str(num_layers) + "\n")
    run(m, net, optimizer, lr, batch_size, epochs, n_points=N_POINTS, n_train=N_TRAIN,data_path=data_path,model_path=model_path ,result_path=result_path, loss_history_path=loss_history_path)
    
    
    if N_TRAIN is None:
      d = np.load(os.path.join(data_path,"train.npz"))
      N_TRAIN, _ = d["y_train"].shape
    if N_POINTS is None:
      d = np.load(os.path.join(data_path,"train.npz"))
      _, N_POINTS = d["y_train"].shape
    
    


if __name__ == "__main__":
    main()
