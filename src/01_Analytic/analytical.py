import numpy as np
from pathlib import Path
from numpy.polynomial.legendre import leggauss

# ============================================================
# 1. Parámetros
# ============================================================
n_models = 70
velocities = np.linspace(1500, 4500, n_models).round().astype(int)
print(velocities)

nx = 100
nz = 100
nt = 460

dh = 7.0       # [m]
dt = 0.001      # [s]
f0 = 15.0      # [Hz]
t0 = 1.5 / f0 # 0.1 equivalente
amplitude = 1.0

x = np.arange(nx) * dh
z = np.arange(nz) * dh
t = np.arange(nt) * dt

# Fuente en una de las dos columnas centrales y a 300 m.
ix_source = nx // 2
iz_source = nz // 2

x_source = x[ix_source]
z_source = z[iz_source]

# Normalizacion
# P_new[ix_source, iz_source] += src[it_new]
source_scale = dh**2 / dt**2

n_quad = 96
nodes, weights = leggauss(n_quad)

output_path = Path(
    "data/wavefields/analytical100/wavefield_59.npy"
)
output_path.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. Ricker causal
# ============================================================
def ricker(time):
    time = np.asarray(time)
    a = (np.pi * f0 * (time - t0))**2
    wavelet = amplitude * (1.0 - 2.0 * a) * np.exp(-a)
    return np.where(time >= 0.0, wavelet, 0.0)

# ============================================================
# 3. Distancias directa e imagen
# ============================================================
# Trabajamos en unidades de píxel para identificar exactamente
# las distancias repetidas.
IZ, IX = np.indices((nz, nx))

distance_squared_direct = (
    (IX - ix_source)**2 + (IZ - iz_source)**2
)
#distance_squared_image = (
#    (IX - ix_source)**2 + (IZ + iz_source)**2
#)

# Una tabla común para ambas contribuciones.
unique_squared, inverse = np.unique(
    np.concatenate([
        distance_squared_direct.ravel(),
        #distance_squared_image.ravel(),
    ]),
    return_inverse=True,
)

n_pixels = nz * nx

index_direct = inverse[:n_pixels].reshape(nz, nx)
#index_image = inverse[n_pixels:].reshape(nz, nx)

distances = dh * np.sqrt(unique_squared.astype(float))

# ============================================================
# 4. Respuesta puntual 2D por distancia
# ============================================================
def response_by_distance(distances, times, c):
    """
    Devuelve un array (Nt, número de distancias).

    Integra la función de Green 2D mediante:
        v = (r/c) * cosh(u)

    La sustitución elimina la singularidad integrable
    del frente de onda para r > 0.
    """
    travel = distances / c

    response = np.zeros(
        (len(times), len(distances)),
        dtype=np.float64,
    )

    # La fuente puntual continua es singular en r = 0.
    singular = (distances == 0.0)
    response[:, singular] = np.nan

    prefactor = source_scale / (2.0 * np.pi * c**2)

    for it, time in enumerate(times):
        active = (travel > 0.0) & (time > travel)

        if not np.any(active):
            continue

        a = travel[active]
        u_max = np.arccosh(time / a)

        u = (
            0.5 * u_max[:, None]
            * (nodes[None, :] + 1.0)
        )

        emission_time = time - a[:, None] * np.cosh(u)

        integral = (
            0.5 * u_max
            * np.sum(
                ricker(emission_time) * weights[None, :],
                axis=1,
            )
        )

        response[it, active] = prefactor * integral

    return response

# ============================================================
# 5. Crear el archivo y calcular los siete cubos
# ============================================================
# Memmap escribe directamente en un archivo .npy válido.
# float32: aproximadamente 110 MB para el cubo completo.
shape = (len(velocities), nt, nz, nx)

wavefields = np.lib.format.open_memmap(
    output_path,
    mode="w+",
    dtype=np.float32,
    shape=shape,
)

for imodel, c in enumerate(velocities):
    print(
        f"Calculando modelo {imodel + 1}/{len(velocities)}: "
        f"c = {c:g} m/s",
        flush=True,
    )

    response = response_by_distance(distances, t, c)

    # Reconstrucción del campo:
    # (Nt, Nz, Nx), directa menos fuente imagen.
    field = (
        response[:, index_direct]
        #- response[:, index_image]
    )

    # Verificar la condición de superficie libre.
    #surface_error = np.max(np.abs(field[:, 0, :]))
    #assert surface_error == 0.0
    
    max_amplitude = np.nanmax(field)

    # Únicamente el nodo de la fuente debe contener NaN.
    assert np.count_nonzero(np.isnan(field)) == nt

    wavefields[imodel] = field.astype(np.float32)
    wavefields.flush()

    print(
        f"  Guardado: {field.shape}; "
        #f"máximo |p| en superficie = {surface_error:.3e}",
        f"maxima amplitud = {max_amplitude:.3e}" ,
        flush=True,
    )

    del response, field

# Cerrar el acceso de escritura.
del wavefields

# ============================================================
# 6. Verificar el archivo guardado
# ============================================================
saved = np.load(output_path, mmap_mode="r")

assert saved.shape == (n_models, nt, nx, nz)
assert saved.dtype == np.float32

print(f"\nArchivo: {output_path}")
print(f"Forma: {saved.shape}")
print(f"Tipo: {saved.dtype}")
print(f"Tiempo: {t[0]:.3f} a {t[-1]:.3f} s")
print(f"Fuente: x={x_source:g} m, z={z_source:g} m")
print(f"Tamaño: {output_path.stat().st_size / 1e6:.1f} MB")