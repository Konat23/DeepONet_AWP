# Sebastian Guerrero 30-Oct
# Funciones utiles que se llaman desde diferentes partes del codigo
# Antes estaban repartidas - Ahora estan centralizadas
import numpy as np
import matplotlib.pyplot as plt


def plot_traces_v_ax(
    ax,
    cube_3d,
    cube_3d2=None,
    z_index=0,
    title=None,
    borde=5,
    n_traces=10,
    spread=None,                 # controla apertura lateral; si None usa Nx/20
    ref_kwargs=None,             # estilo para referencia
    pred_kwargs=None,            # estilo para cube_3d2
    invert_time_axis=True,
    show_grid=True,
    legend=True,
    delta_t = 0.001
):
    """
    Versión "para subplot" de plot_traces_v: dibuja en un Axes (ax) ya existente.

    cube_3d: shape (t, x, z) según tu docstring, PERO tu indexado usa cube_3d[:, Rz, Rx]
             (o sea: tiempo, z, x). Esta función mantiene tu misma convención.
    """
    if ref_kwargs is None:
        ref_kwargs = dict(color="black", linewidth=1.0)
    if pred_kwargs is None:
        pred_kwargs = dict(color="red", linestyle="dashed", alpha=0.7, linewidth=1.0)

    def get_traces(Rx, Rz, cube):
        Nt = cube.shape[0]
        T = np.zeros((Rx.shape[0], Nt), dtype=float)
        for i in range(Rx.shape[0]):
            T[i] = cube[:, Rz[i], Rx[i]]
        return T

    # Dimensiones
    Nt, Nz, Nx = cube_3d.shape
    if cube_3d2 is not None:
        assert cube_3d2.shape == cube_3d.shape, "cube_3d2 debe tener la misma forma que cube_3d"

    # Selección de trazas
    borde = int(max(0, borde))
    if Nx - 2 * borde <= 0:
        raise ValueError(f"borde={borde} es demasiado grande para Nx={Nx}")

    n_traces = int(max(1, n_traces))
    Rx_subset = np.linspace(borde, Nx - borde - 1, n_traces, dtype=int)
    Rz_subset = np.full(Rx_subset.shape[0], z_index, dtype=int)

    T1 = get_traces(Rx_subset, Rz_subset, cube_3d)
    T2 = get_traces(Rx_subset, Rz_subset, cube_3d2) if cube_3d2 is not None else None

    t = np.arange(Nt)*delta_t#*1000 # *1000 para mostrar en ms

    # Factor de escala lateral
    if spread is None:
        spread = Nx / 20.0
    spread = float(spread)

    # Limpiar el axes (opcional; si no quieres, borra esta línea)
    ax.cla()

    # Plot referencia
    for i in range(T1.shape[0]):
        tr = T1[i]
        denom = np.max(np.abs(tr))
        scaled = (tr / denom * spread) if denom > 0 else np.zeros_like(tr)
        offset = Rx_subset[i]
        label = "Ground Truth" if i == 0 else None
        ax.plot(offset + scaled, t, label=label, **ref_kwargs)

    # Plot comparación
    if T2 is not None:
        for i in range(T2.shape[0]):
            tr = T2[i]
            denom = np.max(np.abs(tr))
            scaled = (tr / denom * spread) if denom > 0 else np.zeros_like(tr)
            offset = Rx_subset[i]
            label = "DeepOnet Prediction" if i == 0 else None
            ax.plot(offset + scaled, t, label=label, **pred_kwargs)

    ax.set_xlabel("Receptors (x)")
    ax.set_ylabel("Time (s)")
    ax.set_title(title if title else f"Traces at z={z_index}")

    if invert_time_axis:
        ax.invert_yaxis()

    if show_grid:
        ax.grid(True, alpha=0.3)

    if legend:
        ax.legend(loc="upper right", fontsize="small")

    # Para que el eje x tenga sentido como "receptores"
    ax.set_xlim(-0.5, Nx - 0.5)
    ax.set_ylim(t.max() - delta_t, 0) if invert_time_axis else ax.set_ylim(0, t.max() - delta_t)

    return ax

def multi_plot_propagation(vel_model, ground_truth, prediction, time_1, time_2, z_index, output_file=None, n_traces=10, delta_t=0.001):
    
    
    from matplotlib.gridspec import GridSpec
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize

    # -----------------------------
    # Estilo global
    # -----------------------------
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.titlesize": 13,
    })

    # -----------------------------
    # Datos
    # -----------------------------

    velocity_model = vel_model
    unique_vals = np.unique(vel_model)
    print(f"Unique velocity values: {unique_vals}")
    num_classes = len(unique_vals)
    base_cmap = plt.get_cmap("viridis")
    norm_cont = Normalize(vmin=1500, vmax=4500)
    colors = base_cmap(norm_cont(unique_vals))    
    cmap_velocity = ListedColormap(colors)
    
    norm = BoundaryNorm(
        boundaries=np.append(unique_vals - 0.5, unique_vals[-1] + 0.5),
        ncolors=num_classes,
    )
    
    gt1 = ground_truth[time_1]
    gt2 = ground_truth[time_2]

    pred1 = prediction[time_1]
    pred2 = prediction[time_2]

    err1 = np.abs(gt1 - pred1)
    err2 = np.abs(gt2 - pred2)
    # -----------------------------
    # Figura y Grid
    # -----------------------------
    fig = plt.figure(figsize=(14, 6), constrained_layout=False)

    gs = GridSpec(
        2, 7, figure=fig,
        width_ratios=[1, 1, 1, 0.04, 1, 0.04, 0.04]
    )

    # Subplots
    ax00 = fig.add_subplot(gs[0, 0])
    ax10 = fig.add_subplot(gs[1, 0])

    ax01 = fig.add_subplot(gs[0, 1])
    ax11 = fig.add_subplot(gs[1, 1])

    ax02 = fig.add_subplot(gs[0, 2])
    ax12 = fig.add_subplot(gs[1, 2])

    cax_seismic = fig.add_subplot(gs[:, 3])

    ax03 = fig.add_subplot(gs[0, 4])
    ax13 = fig.add_subplot(gs[1, 4])

    cax_gray = fig.add_subplot(gs[:, 5])

    # -----------------------------
    # Límites compartidos
    # -----------------------------
    #vmin = min(gt1.min(), gt2.min(), pred1.min(), pred2.min())
    vmax = max(gt1.max(), gt2.max(), pred1.max(), pred2.max())
    vmin = -vmax

    # -----------------------------
    # Plots
    # -----------------------------
    im_vel = ax00.imshow(velocity_model, cmap=cmap_velocity, aspect="equal", norm=norm)
    ax00.set_title("Velocity model")

    plot_traces_v_ax(
        ax10,
        cube_3d=ground_truth,
        cube_3d2=prediction,   # o None si no quieres comparar
        z_index=z_index,             # el z que quieras
        title="Shot gather",
        borde=5,
        n_traces=n_traces,
        spread=None,           # o un número para abrir/cerrar más
        delta_t = delta_t
    )
    ax10.set_title(f"Shot gather at z={z_index} Km")

    im_gt1 = ax01.imshow(gt1, cmap="seismic", vmin=vmin, vmax=vmax, aspect="equal")
    im_gt2 = ax11.imshow(gt2, cmap="seismic", vmin=vmin, vmax=vmax, aspect="equal")

    im_pr1 = ax02.imshow(pred1, cmap="seismic", vmin=vmin, vmax=vmax, aspect="equal")
    im_pr2 = ax12.imshow(pred2, cmap="seismic", vmin=vmin, vmax=vmax, aspect="equal")

    # Escala común para ambos errores absolutos
    vmax_error = max(err1.max(), err2.max())

    im_er1 = ax03.imshow(
        err1, cmap="Greys", vmin=0, vmax=vmax_error, aspect="equal"
    )
    im_er2 = ax13.imshow(
        err2, cmap="Greys", vmin=0, vmax=vmax_error, aspect="equal"
    )

    # -----------------------------
    # Títulos de columnas
    # -----------------------------
    ax01.set_title("Ground Truth")
    ax02.set_title("DeepOnet Prediction")
    ax03.set_title("Absolute Error")

    # -----------------------------
    # Colorbars
    # -----------------------------
    # Colorbar del modelo de velocidad (integrado)
    cb_vel = plt.colorbar(im_vel, ax=ax00, ticks=unique_vals,  pad=0.12)
    cb_vel.ax.invert_yaxis()
    #cb_vel = fig.colorbar(im_vel, ax=ax00, fraction=0.046, pad=0.04)
    cb_vel.set_label("Velocity (m/s)")
    cb_vel.ax.yaxis.set_label_position("left")
    
    cb_seismic = fig.colorbar(im_pr1, cax=cax_seismic)
    cb_seismic.set_label("Amplitude", labelpad=10)
    cb_seismic.ax.yaxis.set_label_position("left")
    #cb_seismic.ax.yaxis.tick_left()


    cb_gray = fig.colorbar(im_er1, cax=cax_gray)
    cb_gray.set_label("Absolute error", labelpad=10)
    cb_gray.ax.yaxis.set_label_position("left")

       
    
    fig.text(0.23, 0.75, f"t = {time_1*delta_t} (s)", rotation=90,
         va='center', ha='center', fontsize=14, fontweight='bold')
    
    fig.text(0.23, 0.28, f"t = {time_2*delta_t} (s)", rotation=90,
            va='center', ha='center', fontsize=14, fontweight='bold')
    # -----------------------------
    # Etiquetas de fila (profesional)
    # -----------------------------
    ax02.set_ylabel(f"Depth (Km)")
    ax12.set_ylabel(f"Depth (Km)")
    ax11.set_xlabel(f"offset (Km)")
    ax12.set_xlabel(f"offset (Km)")
    ax13.set_xlabel(f"offset (Km)")


    # -----------------------------
    # Limpieza visual
    # -----------------------------
    for ax in [ax00, ax01, ax02, ax03, ax10, ax11, ax12, ax13]:
        # quitar grid
        ax.grid(False)
        #Poner ticks mas pegados al grafico
        ax.tick_params(axis='both', which='both', length=0)
    # Ajustar layout
    plt.tight_layout()     
    #save pdf
    
    plt.savefig(output_file if output_file else "wave_propagation_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_single_trace(
    ground_truth,
    prediction=None,
    x_index=0,
    z_index=0,
    delta_t=0.001,
    title=None,
    figsize=(10, 5),
    show_grid=True,
    legend=True,
    output_file=None,
):
    """
    Grafica la traza temporal de un punto (x_index, z_index).

    Parámetros
    ----------
    ground_truth : np.ndarray, shape (Nt, Nz, Nx)
        Cubo de referencia.
    prediction : np.ndarray o None, shape (Nt, Nz, Nx)
        Cubo de predicción opcional.
    x_index : int
        Índice espacial en x.
    z_index : int
        Índice espacial en z.
    delta_t : float
        Paso temporal en segundos.
    title : str o None
        Título del gráfico.
    figsize : tuple
        Tamaño de la figura.
    show_grid : bool
        Mostrar cuadrícula.
    legend : bool
        Mostrar leyenda.
    output_file : str o None
        Ruta opcional para guardar la figura.

    Retorna
    -------
    fig, ax : figura y ejes de Matplotlib.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Validar dimensiones e índices
    if ground_truth.ndim != 3:
        raise ValueError("ground_truth debe tener forma (Nt, Nz, Nx).")

    Nt, Nz, Nx = ground_truth.shape

    if not (0 <= x_index < Nx):
        raise IndexError(f"x_index debe estar entre 0 y {Nx - 1}.")

    if not (0 <= z_index < Nz):
        raise IndexError(f"z_index debe estar entre 0 y {Nz - 1}.")

    if prediction is not None and prediction.shape != ground_truth.shape:
        raise ValueError(
            "prediction debe tener la misma forma que ground_truth."
        )

    # Extraer las trazas temporales
    trace_gt = ground_truth[:, z_index, x_index]
    time = np.arange(Nt) * delta_t

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.plot(time, trace_gt, label="Analytical", linewidth=1.5)

    if prediction is not None:
        trace_pred = prediction[:, z_index, x_index]
        ax.plot(time, trace_pred, label="Finite Diference", linestyle="dashed", linewidth=1.5)

    if title is None:
        title = f"Traza temporal en x={x_index}, z={z_index}"

    ax.set_title(title)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")

    if show_grid:
        ax.grid(True, alpha=0.3)

    if legend:
        ax.legend()

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

    return fig, ax

def plot_velocity_and_traces(
    vel_model,
    wavefields,
    labels,
    source_position,
    receiver_position,
    delta_t=0.001,
    dx=None,
    dz=None,
    spatial_unit="m",
    figsize=(15, 5),
    output_file=None,
    show=True,
):
    """
    Grafica el modelo de velocidad y las trazas temporales en un receptor.

    Parámetros
    ----------
    vel_model : np.ndarray, shape (Nz, Nx)
        Modelo de velocidad en m/s.
    wavefields : np.ndarray o lista de np.ndarray
        Uno o varios cubos de forma (Nt, Nz, Nx).
        Todos deben tener la misma forma.
    labels : str o lista de str
        Etiqueta para cada cubo/traza.
    source_position : tuple (x_index, z_index)
        Posición de la fuente en índices de la malla.
    receiver_position : tuple (x_index, z_index)
        Posición del receptor en índices de la malla.
    delta_t : float
        Paso temporal en segundos.
    dx, dz : float o None
        Espaciado espacial. Si se proporcionan ambos, los ejes muestran
        coordenadas físicas; de lo contrario, muestran índices.
    spatial_unit : str
        Unidad de dx y dz, por ejemplo "m" o "km".
    figsize : tuple
        Tamaño de la figura.
    output_file : str o None
        Ruta opcional para guardar la figura.
    show : bool
        Mostrar la figura.

    Retorna
    -------
    fig, (ax_model, ax_traces)
        Figura y los dos ejes principales.
    """
    vel_model = np.asarray(vel_model)

    if vel_model.ndim != 2:
        raise ValueError("vel_model debe tener forma (Nz, Nx).")

    # Permitir un único cubo y un único label.
    if isinstance(wavefields, np.ndarray) and wavefields.ndim == 3:
        wavefields = [wavefields]
    else:
        wavefields = list(wavefields)

    if isinstance(labels, str):
        labels = [labels]
    else:
        labels = list(labels)

    if not wavefields:
        raise ValueError("Debes proporcionar al menos un cubo.")

    if len(wavefields) != len(labels):
        raise ValueError("Debe existir un label por cada cubo.")

    if not np.isfinite(delta_t) or delta_t <= 0:
        raise ValueError("delta_t debe ser positivo y finito.")

    Nz, Nx = vel_model.shape
    wavefields = [np.asarray(cube) for cube in wavefields]

    for cube in wavefields:
        if cube.ndim != 3 or cube.shape[1:] != (Nz, Nx):
            raise ValueError(
                f"Cada cubo debe tener forma (Nt, {Nz}, {Nx})."
            )
        if cube.shape != wavefields[0].shape:
            raise ValueError("Todos los cubos deben tener la misma forma.")

    Nt = wavefields[0].shape[0]
    if Nt == 0:
        raise ValueError("Los cubos deben contener muestras temporales.")

    def validate_position(position, name):
        if len(position) != 2:
            raise ValueError(f"{name} debe ser (x_index, z_index).")

        x, z = position
        if not all(isinstance(i, (int, np.integer)) for i in (x, z)):
            raise TypeError(f"Los índices de {name} deben ser enteros.")

        if not (0 <= x < Nx and 0 <= z < Nz):
            raise IndexError(
                f"{name}: x debe estar entre 0 y {Nx - 1}, "
                f"y z entre 0 y {Nz - 1}."
            )
        return x, z

    xs, zs = validate_position(source_position, "source_position")
    xr, zr = validate_position(receiver_position, "receiver_position")

    # Coordenadas referidas al centro de cada celda.
    if (dx is None) != (dz is None):
        raise ValueError("Proporciona ambos, dx y dz, o ninguno.")

    if dx is None:
        step_x = step_z = 1
        xlabel = "x (índice)"
        ylabel = "z (índice)"
    else:
        if not all(np.isfinite(v) and v > 0 for v in (dx, dz)):
            raise ValueError("dx y dz deben ser positivos y finitos.")
        step_x, step_z = dx, dz
        xlabel = f"x ({spatial_unit})"
        ylabel = f"Profundidad ({spatial_unit})"

    extent = (
        -0.5 * step_x,
        (Nx - 0.5) * step_x,
        (Nz - 0.5) * step_z,
        -0.5 * step_z,
    )

    fig, (ax_model, ax_traces) = plt.subplots(
        1, 2,
        figsize=figsize,
        constrained_layout=True,
    )

    # --------------------------------------------------
    # 1. Modelo de velocidad, fuente y receptor
    # --------------------------------------------------
    im = ax_model.imshow(
        vel_model,
        cmap="viridis",
        origin="upper",
        extent=extent,
        aspect="equal",
    )

    ax_model.scatter(
        xs * step_x, zs * step_z,
        marker="*", s=220,
        color="red", edgecolors="black", linewidths=0.7,
        label="Fuente", zorder=3,
    )

    ax_model.scatter(
        xr * step_x, zr * step_z,
        marker="*", s=220,
        color="yellow", edgecolors="black", linewidths=0.7,
        label="Receptor", zorder=4,
    )

    ax_model.set_title("Modelo de velocidad")
    ax_model.set_xlabel(xlabel)
    ax_model.set_ylabel(ylabel)
    ax_model.legend(loc="best")

    cbar = fig.colorbar(im, ax=ax_model, pad=0.03, shrink=0.85)
    cbar.set_label("Velocidad (m/s)")

    # --------------------------------------------------
    # 2. Trazas temporales en el receptor
    # --------------------------------------------------
    time = np.arange(Nt) * delta_t
    linestyles = ["-", "--", "-.", ":"]

    for i, (cube, label) in enumerate(zip(wavefields, labels)):
        ax_traces.plot(
            time,
            cube[:, zr, xr],
            label=label,
            linewidth=1.5,
            linestyle=linestyles[i % len(linestyles)],
        )

    ax_traces.set_title(f"Receptor: x_index={xr}, z_index={zr}")
    ax_traces.set_xlabel("Tiempo (s)")
    ax_traces.set_ylabel("Amplitud")
    ax_traces.grid(True, alpha=0.3)
    ax_traces.legend()

    if output_file:
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, (ax_model, ax_traces)
  
  