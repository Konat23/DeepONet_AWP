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
    vmax = max(np.nanmax(gt1), 
               np.nanmax(gt2), 
               np.nanmax(pred1), 
               np.nanmax(pred2))
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

    im_er1 = ax03.imshow(err1, cmap="Greys", aspect="equal") 
    im_er2 = ax13.imshow(err2, cmap="Greys", aspect="equal")

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
    label1="ground_truth",
    label2="prediction"
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

    #if prediction is not None and prediction.shape != ground_truth.shape:
    #    raise ValueError(
    #        "prediction debe tener la misma forma que ground_truth."
    #    )

    # Extraer las trazas temporales
    trace_gt = ground_truth[:, z_index, x_index]
    time = np.arange(Nt) * delta_t

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.plot(time, trace_gt, label=label1, linewidth=1.5)

    if prediction is not None:
        trace_pred = prediction[:, z_index, x_index]
        ax.plot(time, trace_pred, label=label2, linestyle="dashed", linewidth=1.5)

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
    from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
    velocities = np.unique(vel_model)
    n_colors = len(velocities)
    
    norm_colors = Normalize(vmin=1500, vmax=4500, clip=True)
    cmap = ListedColormap(
        plt.get_cmap("viridis")(norm_colors(velocities))
    )

    # Límites entre velocidades consecutivas.
    if n_colors > 1:
        boundaries = np.concatenate([
            [velocities[0] - (velocities[1] - velocities[0]) / 2],
            (velocities[:-1] + velocities[1:]) / 2,
            [velocities[-1] + (velocities[-1] - velocities[-2]) / 2],
        ])
    else:
        boundaries = np.array([velocities[0] - 0.5, velocities[0] + 0.5])

    norm = BoundaryNorm(boundaries, ncolors=cmap.N)

    im = ax_model.imshow(
        vel_model,
        cmap=cmap,
        norm=norm,
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

    cbar = fig.colorbar(
        im,
        ax=ax_model,
        boundaries=boundaries,
        ticks=velocities,
        spacing="uniform",  # Todas las secciones tienen igual tamaño.
        drawedges=True,
        pad=0.03,
        shrink=0.85,
    )
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
  
def plot_four_frames(
  cube,
  times=(0.05, 0.10, 0.15, 0.20),
  dt=0.001,
  dh=10.0,
  title="Respuesta al impulso directa",
  output_file=None,
):
  """
  cube: array con forma (nt, nz, nx).
  times: cuatro tiempos solicitados, en segundos.
  dt: intervalo temporal [s].
  dh: espaciamiento de la malla [m].
  """
  cube = np.asarray(cube)

  if cube.ndim != 3:
      raise ValueError("cube debe tener forma (nt, nz, nx).")
  if dt <= 0 or dh <= 0:
      raise ValueError("dt y dh deben ser positivos.")

  times = np.asarray(times, dtype=float)
  if times.shape != (4,) or not np.all(np.isfinite(times)):
      raise ValueError("Debes proporcionar cuatro tiempos finitos.")

  nt, nz, nx = cube.shape
  if np.any(times < 0) or np.any(times > (nt - 1) * dt):
      raise ValueError(
          f"Los tiempos deben estar entre 0 y {(nt - 1) * dt:.3f} s."
      )

  indices = np.rint(times / dt).astype(int)
  frames = cube[indices]

  # Escala simétrica compartida, ignorando NaN e infinitos.
  finite_values = frames[np.isfinite(frames)]
  if finite_values.size == 0:
      raise ValueError("Los frames seleccionados no tienen valores finitos.")

  vmax = np.max(np.abs(finite_values))
  if vmax == 0:
      vmax = 1.0

  cmap = plt.get_cmap("seismic").copy()
  cmap.set_bad("lightgray")

  # Los centros de los píxeles coinciden con x=i*dh, z=j*dh.
  extent = (
      -dh / 2,
      (nx - 0.5) * dh,
      (nz - 0.5) * dh,
      -dh / 2,
  )

  fig, axes = plt.subplots(
      2, 2,
      figsize=(10, 9),
      sharex=True,
      sharey=True,
      layout="constrained",
  )

  for ax, index, frame in zip(axes.flat, indices, frames):
      im = ax.imshow(
          np.ma.masked_invalid(frame),
          cmap=cmap,
          vmin=-vmax,
          vmax=vmax,
          origin="upper",
          extent=extent,
          aspect="equal",
          interpolation="nearest",
      )
      ax.set_title(f"t = {index * dt:.3f} s")
      ax.set_xlabel("x [m]")
      ax.set_ylabel("Profundidad [m]")

  fig.suptitle(title)
  fig.colorbar(
      im,
      ax=axes.ravel().tolist(),
      label="Amplitud",
      shrink=0.85,
      pad=0.03,
  )

  if output_file is not None:
      fig.savefig(output_file, dpi=300, bbox_inches="tight")

  plt.show()
  return fig, axes

def _get_color_limit(slice_2d, center_colorbar, max_value, min_value):
    if max_value is not None and min_value is not None:
        if center_colorbar:
            abs_max = max(abs(min_value), abs(max_value))
            return -abs_max, abs_max
        return min_value, max_value
    else:
        vmin, vmax = slice_2d.min(), slice_2d.max()
        if center_colorbar:
            abs_max = max(abs(vmin), abs(vmax))
            return -abs_max, abs_max
        return vmin, vmax

def show_wave_propagation_movie(
    P,
    output_folder=None,
    step=None,
    dinamic_range=False,
    center_colorbar=True,
    index="test",
):
    """
    Visualizes and optionally saves a movie of wave propagation from a 3D array.
    Parameters:
      P (numpy.ndarray): A 3D array representing the wave propagation data.
                 The first dimension corresponds to time or iterations,
                 and the other two dimensions represent spatial data.
      output_folder (str, optional): Directory where the output images and GIF
                      will be saved. Defaults to "figs".
      step (int, optional): Step size for iterating through the time dimension
                  of the array. Defaults to 50.
      dinamic_range (bool, optional): If True, adjusts the colorbar range dynamically
                      for each frame. Defaults to False.
      center_colorbar (bool, optional): If True, centers the colorbar around zero.
                        Defaults to True.
      save (bool, optional): If True, saves the frames as PNG images and creates
                   a GIF. Defaults to False.
      index (str, optional): Identifier for the model, used in the plot titles
                   and filenames. Defaults to "test".
    Behavior:
      - Displays the wave propagation as a series of frames using matplotlib.
      - Optionally saves each frame as a PNG image and compiles them into a GIF.
    Notes:
      - The function uses the "seismic" colormap for visualization.
      - The `plt.pause(0.1)` is used to control the display duration of each frame.
      - If `save` is True, the function creates a GIF using the `imageio` library.
    Example:
      >>> import numpy as np
      >>> P = np.random.rand(100, 50, 50)  # Example 3D wave data
      >>> show_wave_propagation_movie(P, save=True, index="example")
    """
    if step is None:
        step = P.shape[0] // 100
    if output_folder != None:
        os.makedirs(output_folder, exist_ok=True)
    if not dinamic_range:
        vmin, vmax = (P.min(), P.max())
        print(f"Vmin: {vmin}, Vmax: {vmax}")
        if center_colorbar:
            abs_max = max(abs(vmin), abs(vmax))
            vmin = -abs_max
            vmax = abs_max
    for i in range(0, P.shape[0], step):
        if dinamic_range:
            vmin, vmax = (
                P[i, :, :].min(),
                P[i, :, :].max(),
            )  # Get local min and max for consistent colorbar
            if center_colorbar:
                abs_max = max(abs(vmin), abs(vmax))
                vmin = -abs_max
                vmax = abs_max

        plt.imshow(P[i, :, :], aspect="auto", cmap="seismic", vmin=vmin, vmax=vmax)

        plt.colorbar(label="Amplitude")
        plt.title(f"Model #{index} Iteration {i}")

        if not output_folder is None:
            plt.savefig(f"{output_folder}/iter{i}.png")
        # mostrar el resultado por 0.001 segundos
        plt.pause(0.1)
        plt.clf()  # Clear the figure to avoid overlapping plots
    # Create a gif with the images
    if not output_folder is None:
        import imageio

        images = []
        for i in range(0, P.shape[0], step):
            images.append(imageio.imread(f"{output_folder}/iter{i}.png"))
        imageio.mimsave(f"{output_folder}/wave_propagation.gif", images, duration=0.1)
        print(f"Saved gif to {output_folder}/wave_propagation.gif")


def plot_velmodel(sample, n=None, output_folder=None):
    """
    Plots a discrete speed model using a colormap and saves the plot if an output folder is specified.
    Args:
      n (int): The identifier number for the speed model, used in the plot title and filename.
      sample (numpy.ndarray): A 2D array representing the speed model, where each unique value corresponds to a discrete class.
      output_folder (str, optional): The folder where the plot will be saved as a PNG file. If None, the plot is not saved.
    Returns:
      None
    Notes:
      - The function creates a discrete colormap based on the unique values in the `sample` array.
      - The plot includes a color bar with ticks corresponding to the unique values in the `sample`.
      - The plot is displayed with the origin at the upper-left corner and an aspect ratio of "auto".
      - If `output_folder` is provided, the function ensures the folder exists and saves the plot as "speed_model{n}.png".
    """

    unique_vals = np.unique(sample)
    num_classes = len(unique_vals)

    # Crear un mapa de colores discreto
    cmap = plt.get_cmap("viridis", num_classes)
    norm = BoundaryNorm(
        boundaries=np.append(unique_vals - 0.5, unique_vals[-1] + 0.5),
        ncolors=num_classes,
    )

    plt.figure(figsize=(4, 3))
    im = plt.imshow(sample, cmap=cmap, norm=norm, aspect="equal", origin="upper")

    # Barra de color discreta
    cbar = plt.colorbar(im, ticks=unique_vals)
    cbar.set_label("Velocidad")

    plt.title(f"Modelo de Velocidad #{n}")
    plt.xlabel("Columna")
    plt.ylabel("Capa (Fila)")
    plt.tight_layout()
    if not output_folder is None:
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(f"{output_folder}/speed_model_{n}.png")
        print(f"Saved speed model plot to {output_folder}/speed_model_{n}.png")


def plot_changes(sample, n=None, output_folder=None, vmin=None, vmax=None):
    """
    Plots the magnitude of the Laplacian (second derivatives) of a 2D array.
    Args:
      n (int): Identifier number for the plot, used in the title and filename.
      sample (numpy.ndarray): A 2D array representing the speed model.
      output_folder (str, optional): Folder where the plot will be saved as PNG.
                                     If None, the plot is not saved.
    Returns:
      None
    Notes:
      - Computes the Laplacian using finite differences.
      - Displays the magnitude of the Laplacian with a continuous colormap.
      - If `output_folder` is provided, ensures the folder exists and saves the plot.
    """

    # Derivadas primeras
    dy, dx = np.gradient(sample)

    # Derivadas segundas
    d2y, _ = np.gradient(dy)
    _, d2x = np.gradient(dx)

    # Laplaciano = suma de segundas derivadas
    laplacian = d2x + d2y

    # Magnitud (valor absoluto)
    laplacian_mag = np.abs(laplacian)

    # Graficar
    fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    im0 = axs[0].imshow(
        dy,
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=vmax),
        aspect="equal",
        origin="upper",
    )

    axs[0].set_title(f"Cambio en y #{n}. Max: {np.max(dy):.1f}")
    axs[0].set_xlabel("Columna")
    axs[0].set_ylabel("Capa (Fila)")

    im1 = axs[1].imshow(
        dx,
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=vmax),
        aspect="equal",
        origin="upper",
    )

    axs[1].set_title(f"Cambio en x #{n}. Max: {np.max(dx):.1f}")
    axs[1].set_xlabel("Columna")
    axs[1].set_ylabel("Capa (Fila)")

    fig.colorbar(im0, ax=axs[0], orientation="vertical", label="Gradiente en y")
    fig.colorbar(im1, ax=axs[1], orientation="vertical", label="Gradiente en x")

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(f"{output_folder}/laplacian_{n}.png")
        print(f"Saved Laplacian magnitude plot to {output_folder}/laplacian_{n}.png")

def plot_x_z_t_slices(
    cube_3d,
    x_index,
    z_index,
    t_index,
    cmap="seismic",
    center_colorbar=True,
    max_value=None,
    min_value=None,
    Title=None,
    ouput_folder=None,
    index=None,
    delta_h=1,
    delta_t=1
):
    """
    Genera una figura con 3 subplots: cortes en x, z y t del cubo 3D.

    Parámetros:
    -----------
    cube_3d : numpy.ndarray
        Array 3D con dimensiones (t, x, z)
    x_index : int
        Índice para el corte en la dimensión x
    z_index : int
        Índice para el corte en la dimensión z
    t_index : int
        Índice para el corte en la dimensión t
    cmap : str
        Mapa de colores
    center_colorbar : bool
        Centrar la barra de colores
    max_value : float, opcional
        Valor máximo absoluto para centrar el color
    """
    nt, nz, nx = cube_3d.shape
    if (max_value is not None) ^ (min_value is not None):
        raise ValueError(
            "Debe proporcionar ambos max_value y min_value o ninguno de los dos."
        )
    elif (max_value is None) and (min_value is None):
        max_value = np.max(cube_3d)
        min_value = np.min(cube_3d)
    fig, axs = plt.subplots(1, 3, figsize=(20, 5))

    # X Slice (fijo x => plano t-z)
    x_slice = cube_3d[:, :, x_index].T
    vmin, vmax = _get_color_limit(x_slice, center_colorbar, max_value, min_value)
    im0 = axs[0].imshow(
        x_slice,
        extent=[0, delta_t*nt, delta_h*nz,0],
        aspect=0.5 *(delta_t * nt)/(delta_h * nz),
        cmap=cmap,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    axs[0].set_title(f"Slice en x={x_index*delta_h:.0f}")
    axs[0].set_xlabel("t (seconds)")
    axs[0].set_ylabel("z (meters)")
    fig.colorbar(im0, ax=axs[0])

    # Z Slice (fijo z => plano t-x)
    z_slice = cube_3d[:, z_index, :]
    vmin, vmax = _get_color_limit(z_slice, center_colorbar, max_value, min_value)
    im1 = axs[1].imshow(
        z_slice,
        extent=[0, delta_h*nx, delta_t*nt,0],
        aspect=2 *(delta_h * nx)/(delta_t * nt),
        cmap=cmap,
        origin="upper",
        vmin=vmin,
        vmax=vmax,
    )
    axs[1].set_title(f"Slice en z={z_index*delta_h:.0f}")
    axs[1].set_xlabel("x (meters)")
    axs[1].set_ylabel("t (seconds)")
    fig.colorbar(im1, ax=axs[1])

    # T Slice (fijo t => plano x-z)
    t_slice = cube_3d[t_index, :, :]
    vmin, vmax = _get_color_limit(t_slice, center_colorbar, max_value, min_value)
    im2 = axs[2].imshow(
        t_slice,
        extent=[0, delta_h*nx,  delta_h*nz, 0],
        aspect="equal",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axs[2].set_title(f"Slice en t={t_index*delta_t:.1f} (s)")
    axs[2].set_xlabel("x (meters)")
    axs[2].set_ylabel("z (meters)")
    fig.colorbar(im2, ax=axs[2])
    # Título general
    if Title is not None:
        plt.suptitle(Title)
    plt.tight_layout()
    if ouput_folder is not None:
        os.makedirs(ouput_folder, exist_ok=True)
        plt.savefig(
            f"{ouput_folder}/slices_x{x_index}_z{z_index}_t{t_index}_model_{index}.png"
        )
        print(
            f"Saved slices plot to {ouput_folder}/slices_x{x_index}_z{z_index}_t{t_index}_model_{index}.png"
        )
    plt.show()


def plot_t_slice(
    cube_3d,
    t_index,
    cmap="seismic",
    center_colorbar=True,
    max_value=None,
    min_value=None,
    Title=None,
    ouput_folder=None,
    index=None,
    delta_t = 0.05,
):
    """
    Genera una figura con el corte en t del cubo 3D.

    Parámetros:
    -----------
    cube_3d : numpy.ndarray
        Array 3D con dimensiones (t, x, z)
    t_index : int
        Índice para el corte en la dimensión t
    cmap : str
        Mapa de colores
    center_colorbar : bool
        Centrar la barra de colores
    max_value : float, opcional
        Valor máximo absoluto para centrar el color
    """
    if (max_value is not None) ^ (min_value is not None):
        raise ValueError(
            "Debe proporcionar ambos max_value y min_value o ninguno de los dos."
        )

    fig, ax = plt.subplots(figsize=(2, 3))

    # T Slice (fijo t => plano x-z)
    t_slice = cube_3d[t_index, :, :]
    vmin, vmax = _get_color_limit(t_slice, center_colorbar, max_value, min_value)
    im = ax.imshow(
        t_slice,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(f"Slice en t={t_index*delta_t:.2f}")
    ax.set_xlabel("x (meters)")
    ax.set_ylabel("z (meters)")
    fig.colorbar(im, ax=ax)
    # Título general
    if Title is not None:
        plt.suptitle(Title)
    plt.tight_layout()
    if ouput_folder is not None:
        os.makedirs(ouput_folder, exist_ok=True)
        plt.savefig(f"{ouput_folder}/slice_t{t_index}_model_{index}.png")
        print(
            f"Saved t slice plot to {ouput_folder}/slice_t{t_index}_model_{index}.png"
        )
    plt.show()


def plot_traces(
    cube_3d,
    cube_3d2=None,
    z_index=0,
    x_label="x (meters)",
    y_label="t (seconds)",
    title=None,
    cmap="seismic",
    center_colorbar=True,
    max_value=None,
    borde=5,
):
    """
    Plotea trazas de un array 3D para un índice z específico.

    Parámetros:
    -----------
    cube_3d : numpy.ndarray
      Array 3D con dimensiones (t, x, z)
    cube_3d2 : numpy.ndarray, opcional
      Segundo array 3D con dimensiones (t, x, z) para comparar
    z_index : int
      Índice de la dimensión z que se quiere visualizar
    x_label : str, opcional
      Etiqueta para el eje x (por defecto 'x')
    y_label : str, opcional
      Etiqueta para el eje y (por defecto 't')
    title : str, opcional
      Título del gráfico
    cmap : str, opcional
      Mapa de colores para la visualización (por defecto 'seismic')
    center_colorbar : bool, opcional
      Si centrar la barra de color en cero (por defecto True)
    max_value : float, opcional
      Valor máximo para la escala de color
    borde : int, opcional
      Margen para seleccionar trazas (por defecto 10)
    """

    def get_traces(Rx, Rz, cube_3d):
        """
        Extrae trazas del cubo 3D para las posiciones dadas.
        """
        NTrazas = Rx.shape[0]
        Nt = cube_3d.shape[0]  # Primera dimensión es tiempo

        Trazas = np.zeros([NTrazas, Nt])

        for i in range(NTrazas):
            Trazas[i] = cube_3d[:, Rz[i], Rx[i]]

        return Trazas

    # Dimensiones del cubo
    Nt, Nx, Nz = cube_3d.shape
    if cube_3d2 is not None:
        assert (
            cube_3d2.shape == cube_3d.shape
        ), "cube_3d2 debe tener la misma forma que cube_3d"

    # Todas las trazas en z_index
    Rx_all = np.arange(Nx)
    Rz_all = np.full(Rx_all.shape[0], z_index)
    Trazas_all = get_traces(Rx_all, Rz_all, cube_3d)
    if cube_3d2 is not None:
        Trazas_all2 = get_traces(Rx_all, Rz_all, cube_3d2)

    # Configurar rango de color
    if max_value is not None and center_colorbar:
        raise ValueError("max_value should not be used with center_colorbar=True")

    if max_value is not None:
        vmin, vmax = -max_value, max_value
    else:
        vmin, vmax = Trazas_all.min(), Trazas_all.max()
        if center_colorbar:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max

    Rx_subset = np.linspace(borde, Nx - borde - 1, 10, dtype=int)
    Rz_subset = np.full(Rx_subset.shape[0], z_index)
    Trazas_subset = get_traces(Rx_subset, Rz_subset, cube_3d)
    if cube_3d2 is not None:
        Trazas_subset2 = get_traces(Rx_subset, Rz_subset, cube_3d2)

    plt.figure(figsize=(5, 4))

    # Plot para cube_3d (referencia)
    for i in range(Trazas_subset.shape[0]):
        offset = Rx_subset[i]
        scaled_trace = Trazas_subset[i] / np.max(np.abs(Trazas_subset[i])) * (Nx / 20)
        # Usar el mismo color para todas las trazas de referencia
        # Solo agregar label a la primera traza para evitar duplicados en la leyenda
        label = "Referencia" if i == 0 else ""
        plt.plot(
            scaled_trace + offset, color="black", label=label  # Color fijo para cube_3d
        )

    # Plot para cube_3d2 (DeepOnet) si existe
    if cube_3d2 is not None:
        for i in range(Trazas_subset2.shape[0]):
            offset = Rx_subset[i]
            scaled_trace2 = (
                Trazas_subset2[i] / np.max(np.abs(Trazas_subset2[i])) * (Nx / 20)
            )
            # Solo agregar label a la primera traza para evitar duplicados en la leyenda
            label = "DeepOnet" if i == 0 else ""
            plt.plot(
                scaled_trace2 + offset,
                linestyle="dashed",
                color="red",  # Color fijo para cube_3d2
                alpha=0.7,
                label=label,
            )

    plt.xlabel("tiempo (s)")
    plt.ylabel("Receptores (x)")
    plt.title(title if title else f"Trazas en z={z_index}")

    # Mostrar leyenda
    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_traces_vertical(
    cube_3d,
    cube_3d2=None,
    z_index=0,
    x_label="Receptores (x)",
    y_label="tiempo (s)",
    title=None,
    cmap="seismic",
    center_colorbar=True,
    max_value=None,
    borde=5,
):
    """
    Versión de plot_traces donde las trazas quedan VERTICALES:
    eje X = receptores (x) + amplitud
    eje Y = tiempo (t)
    """

    def get_traces(Rx, Rz, cube_3d):
        NTrazas = Rx.shape[0]
        Nt = cube_3d.shape[0]  # Primera dimensión es tiempo
        Trazas = np.zeros([NTrazas, Nt])
        for i in range(NTrazas):
            Trazas[i] = cube_3d[:, Rz[i], Rx[i]]

        return Trazas

    # Dimensiones del cubo
    Nt, Nz, Nx = cube_3d.shape
    if cube_3d2 is not None:
        assert (
            cube_3d2.shape == cube_3d.shape
        ), "cube_3d2 debe tener la misma forma que cube_3d"

    # Todas las trazas en z_index (solo para rango si lo necesitas)
    Rx_all = np.arange(Nx)
    Rz_all = np.full(Rx_all.shape[0], z_index)
    Trazas_all = get_traces(Rx_all, Rz_all, cube_3d)
    if cube_3d2 is not None:
        Trazas_all2 = get_traces(Rx_all, Rz_all, cube_3d2)

    # Configurar rango de color (aunque aquí no usamos imshow, lo dejo por compatibilidad)
    if max_value is not None and center_colorbar:
        raise ValueError("max_value should not be used with center_colorbar=True")

    if max_value is not None:
        vmin, vmax = -max_value, max_value
    else:
        vmin, vmax = Trazas_all.min(), Trazas_all.max()
        if center_colorbar:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max

    # Selección de algunas trazas para mostrar
    Rx_subset = np.linspace(borde, Nx - borde - 1, 10, dtype=int)
    Rz_subset = np.full(Rx_subset.shape[0], z_index)
    Trazas_subset = get_traces(Rx_subset, Rz_subset, cube_3d)
    if cube_3d2 is not None:
        Trazas_subset2 = get_traces(Rx_subset, Rz_subset, cube_3d2)

    plt.figure(figsize=(5, 5))

    t = np.arange(Nt)  # tiempo en eje vertical

    # --------- PLOT DE cube_3d (Referencia) EN VERTICAL ---------
    for i in range(Trazas_subset.shape[0]):
        offset = Rx_subset[i]  # receptor en eje X
        # Escala la traza en unidades de "receptor" para que no se monten
        scaled_trace = Trazas_subset[i] / np.max(np.abs(Trazas_subset[i])) * (Nx / 20)

        label = "Referencia" if i == 0 else ""
        # Aquí se hace la transposición REAL: x = traza, y = tiempo
        plt.plot(
            scaled_trace + offset,  # eje X
            t,  # eje Y (tiempo)
            color="black",
            label=label,
        )
    plt.gca().invert_yaxis()
    # --------- PLOT DE cube_3d2 (DeepOnet) EN VERTICAL ---------
    if cube_3d2 is not None:
        for i in range(Trazas_subset2.shape[0]):
            offset = Rx_subset[i]
            scaled_trace2 = (
                Trazas_subset2[i] / np.max(np.abs(Trazas_subset2[i])) * (Nx / 20)
            )
            label = "DeepOnet" if i == 0 else ""
            plt.plot(
                scaled_trace2 + offset,
                t,
                linestyle="dashed",
                color="red",
                alpha=0.7,
                label=label,
            )

    plt.xlabel(x_label)  # Receptores (x)
    plt.ylabel(y_label)  # tiempo (s)
    plt.title(title if title else f"Trazas en z={z_index}")

    plt.legend(loc="upper right", fontsize="small")
    plt.grid(True, alpha=0.3)

    # Opcional: si quieres tiempo creciendo hacia abajo (estilo sismograma)
    # plt.gca().invert_yaxis()

    plt.show()

def show_wave_propagation_movie(
    P,
    output_folder=None,
    step=50,
    dinamic_range=False,
    center_colorbar=True,
    index="test",
    idx_img=None,  # New parameter for specific image positions
    max_value=None,
):
    """
    Visualizes and optionally saves a movie of wave propagation from a 3D array.
    Parameters:
      P (numpy.ndarray): A 3D array representing the wave propagation data.
                 The first dimension corresponds to time or iterations,
                 and the other two dimensions represent spatial data.
      output_folder (str, optional): Directory where the output images and GIF
                      will be saved. Defaults to None.
      step (int, optional): Step size for iterating through the time dimension
                  of the array. Defaults to 50.
      dinamic_range (bool, optional): If True, adjusts the colorbar range dynamically
                      for each frame. Defaults to False.
      center_colorbar (bool, optional): If True, centers the colorbar around zero.
                        Defaults to True.
      index (str, optional): Identifier for the model, used in the plot titles
                   and filenames. Defaults to "test".
      idx_img (list, optional): List of relative positions (0-1) indicating which
                   frames to save as images. Defaults to None (no images saved).
    Behavior:
      - Displays the wave propagation as a series of frames using matplotlib.
      - Saves specific frames as PNG images if idx_img is provided.
      - Compiles all displayed frames into a GIF if output_folder is provided.
    Notes:
      - The function uses the "seismic" colormap for visualization.
      - The `plt.pause(0.1)` is used to control the display duration of each frame.
      - GIF is created using the `imageio` library.
    Example:
      >>> import numpy as np
      >>> P = np.random.rand(100, 50, 50)  # Example 3D wave data
      >>> show_wave_propagation_movie(P, output_folder="figs", idx_img=[0, 0.2, 0.5, 0.9])
    """
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    # Determine which frames to save as images
    save_frames = set()
    if idx_img is not None:
        save_frames = {int(idx * (P.shape[0] - 1)) + 1 for idx in idx_img}
    print(save_frames)

    if not dinamic_range:
        vmin, vmax = (P.min(), P.max())
        print(f"Vmin: {vmin}, Vmax: {vmax}")
        if center_colorbar:
            if max_value is not None:
                vmin = -max_value
                vmax = max_value
            else:
                abs_max = max(abs(vmin), abs(vmax))
                vmin = -abs_max
                vmax = abs_max

    images_for_gif = []  # To store all frames for GIF

    for i in range(0, P.shape[0], step):
        if dinamic_range and max_value is not None:
            raise ValueError("max_value should not be used with dinamic_range=True")
        if dinamic_range:
            vmin, vmax = P[i, :, :].min(), P[i, :, :].max()
            if center_colorbar:
                abs_max = max(abs(vmin), abs(vmax))
                vmin, vmax = -abs_max, abs_max

        plt.imshow(P[i, :, :], aspect=1, cmap="seismic", vmin=vmin, vmax=vmax)
        plt.colorbar(label="Amplitude")
        plt.title(f"Model #{index} time t= {i*0.07:.2f}")

        if output_folder is not None:
            if i in save_frames:
                # Guardar imagen PNG solo si está en save_frames
                output_path = f"{output_folder}/iter{i}.png"
                plt.savefig(output_path)
                print(f"Saved image to {output_path}")
            else:
                # Guardar la imagen en un buffer en memoria y leerla con imageio
                buf = BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                images_for_gif.append(imageio.imread(buf))
                buf.close()

        plt.clf()

    # Crear el GIF
    if output_folder is not None and images_for_gif:
        gif_path = f"{output_folder}/wave_propagation_{index}.gif"
        imageio.mimsave(gif_path, images_for_gif, duration=0.1)
        print(f"Saved gif to {gif_path}")

