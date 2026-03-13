import numpy as np
from SALib.sample import sobol_sequence

def compute_cone_mask(z_size=70, x_size=70, t_size=178, t_0=50, t_f=160, r_0=0, r_f=35):
  """
  Compute the indices inside a cone-shaped region.

  Parameters:
  z_size (int): Size of the z-dimension.
  x_size (int): Size of the x-dimension.
  t_size (int): Size of the t-dimension.
  t_0 (int): Start time for the cone.
  t_f (int): End time for the cone.
  r_0 (int): Initial radius of the cone.
  r_f (int): Final radius of the cone.

  Returns:
  list: List of indices (t, z, x) inside the cone.
  """
  in_cone = []
  out_cone = []

  for t in range(t_size):
    R = r_f * (t - t_0) / (t_f - t_0) + r_0  # Radius at time t
    for x in range(x_size):
      for z in range(z_size):
        if ((x - r_f)**2 + (z - r_f)**2 <= R**2) and (t > t_0):
          in_cone.append((t, z, x))
        else:
          out_cone.append((t, z, x))

  return in_cone, out_cone
 
def sobol_cone(wavefield_data, npoints_output):
    """
    Sample points using Sobol sequence with a preference for points inside a cone.

    Parameters:
    wavefield_data (numpy.ndarray): The wavefield data with shape (samples, t, x, z).
    npoints_output (int): Total number of points to sample.

    Returns:
    tuple: 
        - locations (numpy.ndarray): Array of shape (npoints_output, 3) with the sampled points (t, x, z).
        - value (numpy.ndarray): Array of shape (samples, npoints_output) with the wavefield values at the sampled points.
    """
    print("Selecting snapshots using Sobol sequence in coordinate space with cone preference...")
    samples, t_total, x_size, z_size = wavefield_data.shape
    
    # Calculate required points (we'll generate more to account for rejection)
    n_incone = int(npoints_output * 0.9)
    n_outcone = npoints_output - n_incone
    
    # Cone parameters (adjust these to match your cone definition)
    t_0, t_f = 50, 160
    r_0, r_f = 0, 35
    center_x, center_z = r_f, r_f  # Assuming cone is centered at (r_f, r_f)
    
    # Function to check if a point is in the cone
    def is_in_cone(t, x, z):
        if t <= t_0:
            return False
        R = r_f * (t - t_0) / (t_f - t_0) + r_0
        return (x - center_x)**2 + (z - center_z)**2 <= R**2
    
    # Generate Sobol points until we have enough in each region
    in_cone_points = []
    out_cone_points = []
    sobol_dim = 3
    batch_size = max(npoints_output, 1000)  # Generate points in batches for efficiency
    
    print("Generating in-cone Sobol points...")
    while len(in_cone_points) < n_incone:
        # Generate Sobol points in [0,1]^3
        sobol_samples = sobol_sequence.sample(batch_size, sobol_dim)
        
        # Scale to actual coordinates
        t_coords = (sobol_samples[:, 0] * t_total).astype(int)
        x_coords = (sobol_samples[:, 1] * x_size).astype(int)
        z_coords = (sobol_samples[:, 2] * z_size).astype(int)
        
        # Check each point
        for t, x, z in zip(t_coords, x_coords, z_coords):
            if is_in_cone(t, x, z) and len(in_cone_points) < n_incone:
                in_cone_points.append([t, x, z])
    
    print("Generating out-cone Sobol points...")
    while len(out_cone_points) < n_outcone:
        # Generate Sobol points in [0,1]^3
        sobol_samples = sobol_sequence.sample(batch_size, sobol_dim)
        
        # Scale to actual coordinates
        t_coords = (sobol_samples[:, 0] * t_total).astype(int)
        x_coords = (sobol_samples[:, 1] * x_size).astype(int)
        z_coords = (sobol_samples[:, 2] * z_size).astype(int)
        
        # Check each point
        for t, x, z in zip(t_coords, x_coords, z_coords):
            if not is_in_cone(t, x, z) and len(out_cone_points) < n_outcone:
                out_cone_points.append([t, x, z])
    
    # Combine points
    locations = np.array(in_cone_points + out_cone_points)
    
    # Shuffle to mix in-cone and out-cone points
    np.random.shuffle(locations)
    
    # Sample values
    value = np.zeros((samples, npoints_output))
    for i in range(npoints_output):
        t, x, z = locations[i]
        value[:, i] = wavefield_data[:, t, x, z]
    
    print("Sobol cone coordinate sampling completed.")
    print("Selected locations shape:", locations.shape)
    print("Selected values shape:", value.shape)
    
    return locations, value

def random_cone(wavefield_data, npoints_output):
  """
  Randomly sample points from a cone-shaped region in the wavefield data.

  Parameters:
  wavefield_data (numpy.ndarray): The wavefield data with shape (samples, t, x, z).
  npoints_output (int): Total number of points to sample.

  Returns:
  tuple: 
      - locations (numpy.ndarray): Array of shape (npoints_output, 3) with the sampled points (t, x, z).
      - value (numpy.ndarray): Array of shape (samples, npoints_output) with the wavefield values at the sampled points.
  """
  n_outcone = int(npoints_output*0.1)   # 10% of the points will be sampled outside the cone
  n_incone = npoints_output - n_outcone # 90%
  print("Sampling cone")  
  samples, t, x, z = wavefield_data.shape
  # idices inside the cone
  in_cone, out_cone = compute_cone_mask(z_size=z, x_size=x, t_size=t)
  
  # Randomly select locations inside the cone
  in_cone_indices = np.random.randint(0, len(in_cone), size=n_incone)
  out_cone_indices = np.random.randint(0, len(out_cone), size=n_outcone)
  
  print("Getting locations")
  in_cone_locations = np.array([in_cone[i] for i in in_cone_indices])
  out_cone_locations = np.array([out_cone[i] for i in out_cone_indices])
  print("Combining locations")
  locations = np.vstack((in_cone_locations, out_cone_locations))
  value = np.zeros((samples, npoints_output))
  # Shuffle the locations
  print("Shuffling locations")
  np.random.shuffle(locations)
  print("Sampling values")
  for i in range(npoints_output):
    locations[i, 0] = int(locations[i, 0])  # t
    locations[i, 1] = int(locations[i, 1])  # z
    locations[i, 2] = int(locations[i, 2])  # x
    
    value[:, i] = wavefield_data[:, locations[i, 0], locations[i, 1], locations[i, 2]]
  
  print("Sampling completed.")
  print("Sampled locations shape:", locations.shape)
  print("Sampled values shape:", value.shape)
  return locations, value

def random(wavefield_data, npoints_output):
  """
  Randomly sample points from the wavefield data. All samples share the same sampled points.

  Parameters:
  wavefield_data (numpy.ndarray): The wavefield data with shape (samples, t, x, z).
  npoints_output (int): Total number of points to sample.

  Returns:
  tuple: 
      - locations (numpy.ndarray): Array of shape (npoints_output, 3) with the sampled points (t, x, z).
      - value (numpy.ndarray): Array of shape (samples, npoints_output) with the wavefield values at the sampled points.
  """
  samples, t_total, x_size, z_size = wavefield_data.shape
  # Generate random indices for x, z, and t
  t_indices = np.random.randint(0, t_total, size=(npoints_output))
  x_indices = np.random.randint(0, x_size, size=(npoints_output))
  z_indices = np.random.randint(0, z_size, size=(npoints_output))
  
  locations = np.zeros((npoints_output, 3))
  value = np.zeros((samples, npoints_output))
  
  for i in range(npoints_output):
    # Get the random indices
    t_index = t_indices[i]
    x_index = x_indices[i]
    z_index = z_indices[i]
    
    # Fill the locations array
    locations[i, 0] = t_index
    locations[i, 1] = x_index
    locations[i, 2] = z_index
    
    # Fill the values array
    value[:, i] = wavefield_data[:, t_index, x_index, z_index]
    
  print("Sampling completed.")
  print("Sampled locations shape:", locations.shape)
  print("Sampled values shape:", value.shape)
  return locations, value

def sobol(wavefield_data, npoints_output):
    """
    Sample points using Sobol sequence from the wavefield data.

    Parameters:
    wavefield_data (numpy.ndarray): The wavefield data with shape (samples, t, x, z).
    npoints_output (int): Total number of points to sample.

    Returns:
    tuple: 
        - locations (numpy.ndarray): Array of shape (npoints_output, 3) with the sampled points (t, x, z).
        - value (numpy.ndarray): Array of shape (samples, npoints_output) with the wavefield values at the sampled points.
    """
    print("Selecting points using Sobol sequence...")
    samples, t_total, x_size, z_size = wavefield_data.shape
    
    # Generate Sobol sequence indices
    sobol_indices = sobol_sequence.sample(npoints_output, 3)
    
    # Scale indices to fit within the bounds of t, x, z
    t_indices = np.clip((sobol_indices[:, 0] * t_total).astype(int), 0, t_total - 1)
    x_indices = np.clip((sobol_indices[:, 1] * x_size).astype(int), 0, x_size - 1)
    z_indices = np.clip((sobol_indices[:, 2] * z_size).astype(int), 0, z_size - 1)
    
    # Initialize output arrays
    locations = np.zeros((npoints_output, 3))
    value = np.zeros((samples, npoints_output))
    
    # Fill locations and values
    for i in range(npoints_output):
        locations[i] = [t_indices[i], x_indices[i], z_indices[i]]
        value[:, i] = wavefield_data[:, t_indices[i], x_indices[i], z_indices[i]]
    
    print("Sobol selection completed.")
    print("Selected locations shape:", locations.shape)
    print("Selected values shape:", value.shape)
    
    return locations, value

def select_snapshots(wavefield_data, index_snapshots):
    """
    Select all points from specific time indices in the wavefield data.

    Parameters:
    wavefield_data (numpy.ndarray): The wavefield data with shape (samples, t, x, z).
    index_snapshots (list): List of time indices to select.

    Returns:
    tuple: 
        - locations (numpy.ndarray): Array of shape (npoints_output, 3) with the sampled points (t, x, z).
        - value (numpy.ndarray): Array of shape (samples, npoints_output) with the wavefield values at the sampled points.
    """
    print("Selecting snapshots...")
    samples, t_total, x_size, z_size = wavefield_data.shape
    n_times = len(index_snapshots)
    npoints_output = n_times * x_size * z_size
    
    # Initialize output arrays
    locations = np.zeros((npoints_output, 3))
    value = np.zeros((samples, npoints_output))
    
    point_idx = 0
    for t_idx in index_snapshots:
        for x_idx in range(x_size):
            for z_idx in range(z_size):
                # Fill locations array
                locations[point_idx, 0] = t_idx
                locations[point_idx, 1] = x_idx
                locations[point_idx, 2] = z_idx
                
                # Fill values array
                value[:, point_idx] = wavefield_data[:, t_idx, x_idx, z_idx]
                
                point_idx += 1
                
    print("Snapshot selection completed.")
    print("Selected locations shape:", locations.shape)
    print("Selected values shape:", value.shape)
    return locations, value
