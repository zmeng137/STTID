import torch
import numpy as np
from scipy.stats import powerlaw
import matplotlib.pyplot as plt

def write_to_tns_fast_numpy(coo_array, file_path):    
    output_data = np.column_stack((coo_array.indices().T, coo_array.values()))  # Combine coordinates and data into a single array
    np.savetxt(file_path, output_data, fmt='%d '*coo_array.indices().shape[0] + '%.16f')  # Use numpy's savetxt which is optimized for batch writing

def generate_sparse_tensor_powerlaw(shape, density=0.1, alpha=2.0, seed=None):
    """
    Generate a random sparse tensor with power law distributed values.
    
    Args:
        shape: Tuple defining tensor dimensions (e.g., (100, 100) for 2D)
        density: Fraction of non-zero elements (0 < density <= 1)
        alpha: Power law exponent (alpha > 1 for finite mean)
        seed: Random seed for reproducibility
    
    Returns:
        torch.sparse_coo_tensor: Sparse tensor with power law distributed values
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Calculate total number of elements and non-zero elements
    total_elements = np.prod(shape)
    nnz = int(total_elements * density)
    
    # Generate random indices for non-zero elements
    indices = []
    for dim_size in shape:
        idx = torch.randint(0, dim_size, (nnz,))
        indices.append(idx)
    
    indices = torch.stack(indices)
    
    # Generate power law distributed values
    # scipy's powerlaw uses a different parameterization: powerlaw(a) where a = alpha - 1
    power_law_values = powerlaw.rvs(alpha - 1, size=nnz)
    
    # Convert to torch tensor
    values = torch.from_numpy(power_law_values).float()
    
    # Create sparse COO tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    
    # Coalesce to combine duplicate indices
    sparse_tensor = sparse_tensor.coalesce()
    
    return sparse_tensor

def generate_sparse_tensor_powerlaw_custom(shape, density=0.1, alpha=2.0, 
                                         xmin=1.0, seed=None):
    """
    Alternative implementation with custom power law generation.
    
    Args:
        shape: Tuple defining tensor dimensions
        density: Fraction of non-zero elements
        alpha: Power law exponent
        xmin: Minimum value for power law distribution
        seed: Random seed for reproducibility
    
    Returns:
        torch.sparse_coo_tensor: Sparse tensor with power law distributed values
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    total_elements = np.prod(shape)
    nnz = int(total_elements * density)
    
    # Generate random indices
    indices = []
    for dim_size in shape:
        idx = torch.randint(0, dim_size, (nnz,))
        indices.append(idx)
    
    indices = torch.stack(indices)
    
    # Generate power law values using inverse transform sampling
    # For power law: P(X >= x) = (x/xmin)^(-alpha+1)
    # Inverse: x = xmin * (1-u)^(-1/(alpha-1)) where u ~ Uniform(0,1)
    u = np.random.uniform(0, 1, nnz)
    power_law_values = xmin * (1 - u) ** (-1 / (alpha - 1))
    
    values = torch.from_numpy(power_law_values).float()
    
    # Create and return sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)
    return sparse_tensor.coalesce()

def visualize_distribution(sparse_tensor, bins=50):
    """
    Visualize the distribution of non-zero values in the sparse tensor.
    
    Args:
        sparse_tensor: The sparse tensor to analyze
        bins: Number of histogram bins
    """
    values = sparse_tensor.values().numpy()
    
    plt.figure(figsize=(12, 4))
    
    # Linear scale histogram
    plt.subplot(1, 2, 1)
    plt.hist(values, bins=bins, alpha=0.7, density=True)
    plt.title('Power Law Distribution (Linear Scale)')
    plt.xlabel('Value')
    plt.ylabel('Density')
    
    # Log-log scale
    plt.subplot(1, 2, 2)
    plt.hist(values, bins=bins, alpha=0.7, density=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Power Law Distribution (Log-Log Scale)')
    plt.xlabel('Value (log scale)')
    plt.ylabel('Density (log scale)')
    
    plt.tight_layout()
    plt.savefig("randTensor.png")

# Example usage
if __name__ == "__main__":
    # Generate a 2D sparse tensor
    shape = (1000, 1000)
    density = 0.01  # 1% of elements are non-zero
    alpha = 2.5     # Power law exponent
    
    print(f"Generating sparse tensor of shape {shape} with {density*100}% density")
    print(f"Power law exponent: {alpha}")
    
    # Method 1: Using scipy.stats.powerlaw
    sparse_tensor_1 = generate_sparse_tensor_powerlaw(
        shape=shape, 
        density=density, 
        alpha=alpha, 
        seed=42
    )
    
    print(f"\nMethod 1 - Sparse tensor statistics:")
    print(f"Shape: {sparse_tensor_1.shape}")
    print(f"Number of stored values: {sparse_tensor_1._nnz()}")
    print(f"Actual density: {sparse_tensor_1._nnz() / np.prod(shape):.4f}")
    print(f"Value range: [{sparse_tensor_1.values().min():.4f}, {sparse_tensor_1.values().max():.4f}]")
    print(f"Mean value: {sparse_tensor_1.values().mean():.4f}")
    
    # Method 2: Custom power law generation
    sparse_tensor_2 = generate_sparse_tensor_powerlaw_custom(
        shape=shape,
        density=density,
        alpha=alpha,
        xmin=1.0,
        seed=42
    )
    
    print(f"\nMethod 2 - Sparse tensor statistics:")
    print(f"Shape: {sparse_tensor_2.shape}")
    print(f"Number of stored values: {sparse_tensor_2._nnz()}")
    print(f"Value range: [{sparse_tensor_2.values().min():.4f}, {sparse_tensor_2.values().max():.4f}]")
    print(f"Mean value: {sparse_tensor_2.values().mean():.4f}")
    
    # Convert to dense for small example
    if np.prod(shape) <= 100:
        print(f"\nDense representation:\n{sparse_tensor_1.to_dense()}")
    
    # Visualize distribution (uncomment to see plots)
    visualize_distribution(sparse_tensor_1)
    
    # Example operations with sparse tensors
    print(f"\nExample operations:")
    print(f"Sum of all values: {torch.sparse.sum(sparse_tensor_1).item():.4f}")
    print(f"Number of dimensions: {sparse_tensor_1.dim()}")
    
    # For 3D tensor example
    tensor_4d = generate_sparse_tensor_powerlaw(
        shape=(10, 10, 10, 10),
        density=0.01,
        alpha=3.0,
        seed=100
    )
    print(f"\n3D tensor shape: {tensor_4d.shape}")
    print(f"3D tensor non-zero elements: {tensor_4d._nnz()}")
    
    outputPath = "powerlaw_test_4d.tns" 
    write_to_tns_fast_numpy(tensor_4d, outputPath)
    pass