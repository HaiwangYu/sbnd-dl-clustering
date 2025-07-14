import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
def get_isnu_labels(truth_file, g2f_file, max_distance=5):
    """
    Extract labels from truth data for points in the g2f file using nearest neighbor matching.
    
    Parameters:
    -----------
    truth_file : str
        Path to the truth data JSON file
    g2f_file : str
        Path to the G2F NPZ file
    max_distance : float, optional
        Maximum allowed distance for matching, default is 3
        
    Returns:
    --------
    np.array
        Array of isnu labels for each point in the g2f file
    """
    # Load the truth file
    with open(truth_file, 'r') as f:
        truth_data = json.load(f)
    
    # Load the G2F file
    g2f_data = np.load(g2f_file)
    points = g2f_data['points']
    
    # Extract coordinates from points (with normalization)
    x = points[:, 0]/10.
    y = points[:, 1]/10.
    z = points[:, 2]/10.
    points_coords = np.array(list(zip(x, y, z)))
    
    # Extract x, y, z coordinates from truth_data
    truth_coords = np.array(list(zip(truth_data['x'], truth_data['y'], truth_data['z'])))
    
    # Create KNN model for truth data
    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(truth_coords)
    
    # Find closest point in truth data for each point in points data
    distances, indices = knn.kneighbors(points_coords)
    
    # For each point in points, get the truth_data 'q' value of its nearest neighbor
    # If distance > max_distance, label as -2
    # -1: for all truth points that cannot be backtracked
    # 0: has a backtracked truth point, but not a kBeamNeutrino
    # 1: has a backtracked truth point and is a kBeamNeutrino
    # -2: the KDTree search did not find a point within max_distance
    isnu = np.array([-2 if dist > max_distance else truth_data['q'][idx] 
                    for dist, idx in zip(distances.flatten(), indices.flatten())])
    
    return isnu
