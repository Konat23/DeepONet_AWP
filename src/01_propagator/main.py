# -*- coding: utf-8 -*-
"""

@author: Sebastian Guerrero
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import os
from src import propagator
from src.myutils import find_files, timing, expand_edges_speedwave
from tqdm import tqdm

sys.path.append( os.path.dirname(os.path.dirname(__file__)))
from constants import DATA_NAME

ID_LIST = [0,1,2,3,59] # selecte models files
N_MODELs = None#None #modelos por paquete, None -> todos (500)
data_set = DATA_NAME
VELMODEL_PATH = os.path.join("data", "vel_models", data_set) 
SOURCE_PATH = os.path.join("data","source_functions")
#output
WAVEFIELD_PATH = os.path.join("data", "wavefields", data_set+"_multisource")

data_type = np.float32


class WavePropagationSimulator:
    def __init__(self, dt, tEnd, dz, dx, frec, CPMLimit, FreeSurfaceLimit):
        self.dt = dt
        self.tEnd = tEnd
        self.Nt = int(tEnd / dt)
        self.t = np.linspace(0, tEnd - dt, self.Nt)
        self.dz = dz
        self.dx = dx
        self.frec = frec
        self.g = self._generate_source()
        self.Pt = None
        self.P = None
        self.d2P_dt2 = None
        self.CPMLimit = CPMLimit
        self.FreeSurfaceLimit = FreeSurfaceLimit

    def _generate_source(self):
        a = (np.pi * self.frec) ** 2
        self.t0 = 0.1
        d_ricker = (-2 * a * (self.t - self.t0) * np.exp(-a * (self.t - self.t0) ** 2)).T
        ricker = ((1 - 2 * a * (self.t - self.t0) ** 2) * np.exp(-a * (self.t - self.t0) ** 2)).T
        source = ricker
        os.makedirs(SOURCE_PATH, exist_ok=True)
        np.save(os.path.join(SOURCE_PATH, "ricker.npy"),np.column_stack((self.t,source)))
        return source
        #return np.load(SOURCE_PATH)[:, 1]

    def set_speed_model(self, vp):
        self.vp = vp
        self.Nz = vp.shape[1]
        self.Nx = vp.shape[0]
    
    def set_source_locations(self, Ix0, Iz0):
        self.Ix0 = Ix0
        self.Iz0 = Iz0

    # @timing
    def compute_wave_propagation(self):
        self.Pt, self.P, self.d2P_dt2 = propagator.propagator(
            self.vp,
            self.g,
            self.Ix0,
            self.Iz0,
            self.dx,
            self.dz,
            self.dt,
            10e3,
            self.frec,
            self.CPMLimit,
        )
        self.P = np.transpose(self.P, (2, 1, 0))
        # Recortar las dimensiones de la matriz P si CPMLimit es mayor que 0
        if self.CPMLimit > 0:
            self.P = self.P[:, : -self.CPMLimit, self.CPMLimit : -self.CPMLimit]
        if self.FreeSurfaceLimit > 0:
            self.P = self.P[:, self.FreeSurfaceLimit :, :]
        # Convertir a tipo de dato especificado
        self.P = self.P.astype(data_type)

    def save_metadata(self, output_folder="wave"):
        os.makedirs(output_folder, exist_ok=True)
        text_file = os.path.join(output_folder, "metadata.txt")
        with open(text_file, "w") as f:
            f.write(f"Wave Propagation Simulation Metadata {output_folder}\n")
            f.write(f"date-time: {np.datetime64('now')}\n")
            f.write(f"author: S. Guerrero\n")
            f.write(f"dt: {self.dt}\n")
            f.write(f"tEnd: {self.tEnd}\n")
            f.write(f"dz: {self.dz}\n")
            f.write(f"dx: {self.dx}\n")
            f.write(f"Ix0: {self.Ix0}\n")
            f.write(f"Iz0: {self.Iz0}\n")
            f.write(f"t0: {self.t0}\n")
            f.write(f"frec: {self.frec}\n")
            f.write(f"CPMLimit: {self.CPMLimit}\n")
            f.write(f"Free-Surface: {self.FreeSurfaceLimit}")
            f.write(f"Data type: {self.P.dtype}\n")
            f.write(f"Wave data shape: {self.P.shape}\n")
            f.write(f"Wave Storage space in MB: {self.P.nbytes/1e6} MB\n")

    def load_wave(self, input_folder="wave"):
        self.P = np.load(os.path.join(input_folder, "wavefield.npy"))
        self.Pt = np.load(os.path.join(input_folder, "wavefield_surface.npy"))
        self.d2P_dt2 = np.load(os.path.join(input_folder, "wavefield_full.npy"))
        self.Nt = self.P.shape[-1]

    def plot_source(self):
        plt.plot(self.t, self.g)
        plt.title("Source Time Function")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()
        plt.savefig("source_time_function.png")


def plot_model(n, sample, output_folder="vel_models"):
    os.makedirs(output_folder, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.imshow(sample, cmap="seismic", origin="upper")  # Change origin to "upper"
    plt.colorbar(label="Velocidad (m/s)")
    plt.title(f"Muestra {n} de model1.npy")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.savefig(f"{output_folder}/speed_model{n}.png")
    plt.show()
    # plt.close()


if __name__ == "__main__":
    os.makedirs(WAVEFIELD_PATH, exist_ok=True)
    data_files = sorted(find_files(VELMODEL_PATH, ".npy"))
    data_n = [
        int(os.path.basename(data_file).split("_")[1].split(".")[0])
        for data_file in data_files
    ]
    # print paths 
    print(f"Velocity model path: {VELMODEL_PATH}")
    print(f"Wavefield output path: {WAVEFIELD_PATH}")
    print("Vel Models readed:")
    print(data_n)
    
    # Hyperparameters        
    nx = 100
    nz = 100
    dt = 0.001
    tEnd = 0.46
    dh = 7
    frec = 15
    CPMLimit = 2
    FreeSurfaceLimit = int(np.ceil(861/dh))

    
    # Source location
    Ix0 = np.arange(0, nz, 10) + 5
    Iz0 = np.ones_like(Ix0)
    
    print(f"Source locations: Ix0={Ix0}, Iz0={Iz0}")
    Ix0_pad = Ix0 // 2 + CPMLimit
    Iz0_pad = Iz0 // 2 + FreeSurfaceLimit
    simulator = WavePropagationSimulator(
        dt=dt,
        tEnd=tEnd,
        dz=dh,
        dx=dh,
        frec=frec,
        CPMLimit=CPMLimit,
        FreeSurfaceLimit=FreeSurfaceLimit,
    )

    for id in tqdm(ID_LIST, desc="Processing IDs"):
        if id not in data_n:
            print(f"WARNING: id {id:02d} is not in {VELMODEL_PATH}")
            continue
        data_file = f"model_{id:02d}.npy"
        sw_data = np.load(os.path.join(VELMODEL_PATH, data_file))
        paquete = []
        for idx in tqdm(range(sw_data[:N_MODELs].shape[0]), desc=f"Processing ID {id:02d}"):
            speed_wave = sw_data[idx, 0, :, :]
            xspeed_wave = expand_edges_speedwave(speed_wave, CPMLimit, FreeSurfaceLimit)
            simulator.set_speed_model(xspeed_wave)

            propagaciones_por_fuente = []
            for ix0, iz0 in zip(Ix0_pad, Iz0_pad):
                simulator.set_source_locations(ix0, iz0)
                simulator.compute_wave_propagation()
                propagaciones_por_fuente.append(simulator.P.copy())

            paquete.append(propagaciones_por_fuente)

        paquete = np.stack(paquete, axis=1)
        print(f"Paquete shape: {paquete.shape}")
        np.save(os.path.join(WAVEFIELD_PATH, f"wavefield_{id:02d}.npy"), paquete)
        print(f"Saved wavefield_{id:02d}.npy")
    simulator.save_metadata(WAVEFIELD_PATH)
