import os
import torch
import sparse
import numpy as np
import tensorly as tl
import scipy as sp
from scipy.stats import pareto

# Synthetic Tensor-train Data Generation
# Methodology reference:
#  [1] Discovering faster matrix multiplication algorithms with reinforcement learning
#  [2] SimTensor: A synthetic tensor data generator

# Fast writting using numpy
def write_to_tns_fast_numpy(coo_array, file_path):    
    output_data = np.column_stack((coo_array.coords.T, coo_array.data))  # Combine coordinates and data into a single array
    np.savetxt(file_path, output_data, fmt='%d '*coo_array.coords.shape[0] + '%.16f')  # Use numpy's savetxt which is optimized for batch writing

def generate_sparse_tensor_pareto(shape, density, alpha, seed):
    #Generate a random sparse tensor with power law distributed values.
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
    power_law_values = pareto.rvs(alpha , size=nnz)

    values = torch.from_numpy(power_law_values).float()      # Convert to torch tensor
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape)      # Create sparse COO tensor
    
    # Coalesce to combine duplicate indices
    sparse_tensor = sparse_tensor.coalesce()
    
    return sparse_tensor

# Global parameters
FILENAME = "Syn1.tns"
SHAPE= [10, 10, 10, 10]
RANK = [1, 9, 40, 8, 1]
DENSITY = [1E-1, 1E-2, 1E-2, 1E-1]
SEED = [1, 2, 3, 4]
RVS = sp.stats.norm(loc=5, scale=10).rvs

# Generate random sparse factors
print("Generating sparse factors ...")
dim = len(SHAPE)
TT_factors = []
for i in range(dim):
    shape = [RANK[i], SHAPE[i], RANK[i+1]]
    #F = sparse.random(shape=shape, density=DENSITY[i], random_state=SEED[i], data_rvs=RVS)
    F = generate_sparse_tensor_pareto(shape, DENSITY[i], 3, SEED[i])
    TT_factors.append(F)
print("Generation finishes.")

# Tensor-train factors to tensor
print("Contracting tensor train to tensor ...")
Tensor = tl.tt_to_tensor(TT_factors)
print("Contraction finishes.")
print(f"Tensor shape: {Tensor.shape}; Tensor nnz: {Tensor.nnz}; Tensor density: {Tensor.density}")

# Noise 
NOISE_LEVEL = 0
NOISE_SEED = 0
data = Tensor.data
data_range = np.ptp(data)
nrvs = sp.stats.norm(loc = 0, scale = NOISE_LEVEL * data_range).rvs
noise = sparse.random(shape=[1, len(data)], density = NOISE_LEVEL, random_state = NOISE_SEED, data_rvs = nrvs)
dnoise = noise.todense()[0]
data += dnoise

# Output as tns
print("Writing data ...")
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, FILENAME)
write_to_tns_fast_numpy(Tensor, file_path)
print("Done.")