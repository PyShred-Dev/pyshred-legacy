import matplotlib.pyplot as plt


def forward_backward_walk(start, end, timesteps, forward_first=True):
    """
    Generates a list of tuples where each tuple contains one integer,
    alternating back and forth from `start` to `end` or `end` to `start`
    based on the `forward_first` flag until the list reaches the specified `timesteps`.

    Args:
        start (int): The starting value of the range.
        end (int): The ending value of the range.
        timesteps (int): The desired length of the list.
        forward_first (bool): If True, starts from `start` to `end` first. 
                              If False, starts from `end` to `start`.

    Returns:
        list: A list of tuples containing one integer each.
    """
    indices = []
    
    # Generate alternating indices as tuples until the required length is reached
    while len(indices) < timesteps:
        if forward_first:
            indices += [(i,) for i in range(start, end + 1)]  # Forward direction
            indices += [(i,) for i in range(end, start - 1, -1)]  # Backward direction
        else:
            indices += [(i,) for i in range(end-1, start, -1)]  # Backward direction
            indices += [(i,) for i in range(start, end)]  # Forward direction
    
    # Truncate to the specified length
    indices = indices[:timesteps]
    return indices



def perimeter_walk(height, width, timesteps, clockwise = True):
    # Initialize the grid and starting position
    x, y = 0, 0  # Start at the top-left corner
    
    # Define the directions: right, down, left, up
    if clockwise:
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    else:
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    current_direction = 0    
    path = [(x, y)]
    steps = 1
    
    while steps < timesteps:
        # Get the next position based on the current direction
        dx, dy = directions[current_direction]
        new_x, new_y = x + dx, y + dy
        
        # Check if we are out of bounds, change direction if needed
        if new_x < 0 or new_x >= height or new_y < 0 or new_y >= width:
            current_direction = (current_direction + 1) % 4
            continue
        
        # Move to the new position
        x, y = new_x, new_y
        path.append((x, y))
        steps += 1
        
        # If we hit the corner, switch direction
        if clockwise:
            if (x == 0 and y == width-1) or (x == height-1 and y == width-1) or (x == height-1 and y == 0):
                current_direction = (current_direction + 1) % 4
        else:
            if (x == 0 and y == 0) or (x == height-1 and y == 0) or (x == height-1 and y == width-1):
                current_direction = (current_direction + 1) % 4
    return path




def plot_recon_truth_2D_spatial(data_recon, data_truth, lags, recon_timestep):
    """
    Plots a side-by-side comparison of reconstruction and truth data with a consistent color scale.
    
    Parameters:
    - data_recon: 3D numpy array of reconstruction data (shape: [height, width, timesteps])
    - data_truth: 3D numpy array of truth data (shape: [height, width, timesteps])
    - lags: Number of time steps to offset the truth data
    - recon_timestep: The specific time step to visualize
    """
    # Reconstruction sample
    data_recon_sample = data_recon[:, :, recon_timestep]

    # Truth sample (-1 to have truth be at the same timestep as recon)
    data_truth_sample = data_truth[:, :, recon_timestep + lags - 1]

    # Compute the global min and max for consistent color scale
    vmin = min(data_recon_sample.min(), data_truth_sample.min())
    vmax = max(data_recon_sample.max(), data_truth_sample.max())

    # Create subplots for side-by-side plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot the reconstruction
    im1 = axs[0].imshow(data_recon_sample, cmap='RdBu_r', interpolation='bilinear', vmin=vmin, vmax=vmax)
    axs[0].set_title('Reconstruction')
    plt.colorbar(im1, ax=axs[0])  # Colorbar for the first subplot

    # Plot the truth
    im2 = axs[1].imshow(data_truth_sample, cmap='RdBu_r', interpolation='bilinear', vmin=vmin, vmax=vmax)
    axs[1].set_title('Truth')
    plt.colorbar(im2, ax=axs[1])  # Colorbar for the second subplot

    # Display the plots
    plt.tight_layout()
    plt.show()


def plot_forecast_truth_2D_spatial(data_forecast, data_truth, forecast_timestep):
    """
    Plots a side-by-side comparison of reconstruction and truth data with a consistent color scale.
    
    Parameters:
    - data_recon: 3D numpy array of reconstruction data (shape: [height, width, timesteps])
    - data_truth: 3D numpy array of truth data (shape: [height, width, timesteps])
    - lags: Number of time steps to offset the truth data
    - recon_timestep: The specific time step to visualize
    """
    # Reconstruction sample
    data_predict_sample = data_forecast[:, :, forecast_timestep]

    # Truth sample (-1 to have truth be at the same timestep as recon)
    data_truth_sample = data_truth[:, :, forecast_timestep]

    # Compute the global min and max for consistent color scale
    # vmin = min(data_predict_sample.min(), data_truth_sample.min())
    # vmax = max(data_predict_sample.max(), data_truth_sample.max())

    # Create subplots for side-by-side plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot the reconstruction
    # im1 = axs[0].imshow(data_predict_sample, cmap='RdBu_r', interpolation='bilinear', vmin=vmin, vmax=vmax)
    im1 = axs[0].imshow(data_predict_sample, cmap='RdBu_r', interpolation='bilinear')
    axs[0].set_title('Forecast')
    plt.colorbar(im1, ax=axs[0])  # Colorbar for the first subplot

    # Plot the truth
    # im2 = axs[1].imshow(data_truth_sample, cmap='RdBu_r', interpolation='bilinear', vmin=vmin, vmax=vmax)
    im2 = axs[1].imshow(data_truth_sample, cmap='RdBu_r', interpolation='bilinear')
    axs[1].set_title('Truth')
    plt.colorbar(im2, ax=axs[1])  # Colorbar for the second subplot

    # Display the plots
    plt.tight_layout()
    plt.show()


def plot_recon_vs_truth(recon, truth):
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    # Plot the first array recon
    im1 = axs[0].imshow(recon, cmap='RdBu_r', interpolation='bilinear')  # You can change the colormap (cmap) if you want
    axs[0].set_title('recon')
    plt.colorbar(im1, ax=axs[0])  # Colorbar for the first subplot

    # Plot the second array truth
    im2 = axs[1].imshow(truth, cmap='RdBu_r', interpolation='bilinear')  # You can change the colormap (cmap) if you want
    axs[1].set_title('truth')
    plt.colorbar(im2, ax=axs[1])  # Colorbar for the second subplot

    # Display the plots
    plt.tight_layout()
    plt.show()