import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize

import imageio
from io import BytesIO

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

