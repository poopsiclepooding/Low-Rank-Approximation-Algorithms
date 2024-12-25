import numpy as np

def generate_fast_decay_matrix(m, n, decay_type='exponential', decay_rate=0.1, noise_level=1e-10):
    """
    Generate a matrix with rapidly decaying singular values.
    
    Parameters:
        m (int): Number of rows
        n (int): Number of columns
        decay_type (str): Type of decay - 'exponential', 'polynomial', or 'custom'
        decay_rate (float): Rate of decay
        noise_level (float): Amount of noise to add
        
    Returns:
        A (numpy.ndarray): Matrix with fast decaying spectrum
        true_sigmas (numpy.ndarray): True singular values used
    """
    rank = min(m, n)
    
    # Generate singular values with different decay patterns
    if decay_type == 'exponential':
        # Exponential decay: σᵢ = exp(-i * decay_rate)
        singular_values = np.exp(-np.arange(rank) * decay_rate)
        
    elif decay_type == 'polynomial':
        # Polynomial decay: σᵢ = 1/(i + 1)^decay_rate
        singular_values = 1 / (np.arange(rank) + 1) ** decay_rate
        
    elif decay_type == 'custom':
        # Custom decay: σᵢ = 1/(exp(i * decay_rate) + i^2)
        singular_values = 1 / (np.exp(np.arange(rank) * decay_rate) + 
                             np.arange(rank)**2)
    
    # Normalize singular values
    singular_values = singular_values / singular_values[0]
    
    # Generate random orthonormal matrices
    U = np.linalg.qr(np.random.randn(m, rank))[0]
    V = np.linalg.qr(np.random.randn(n, rank))[0]
    
    # Construct the matrix
    A = U @ np.diag(singular_values) @ V.T
    
    # Add small noise
    if noise_level > 0:
        A += noise_level * np.random.randn(m, n)
    
    return A, singular_values