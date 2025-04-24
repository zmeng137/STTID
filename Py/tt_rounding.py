import numpy as np
import tensorly as tl

def tt_rounding(tensor_train, epsilon):
    # Initialization
    A = tl.tt_to_tensor(tensor_train)  # Reconstruct the tensor from cores
    shape = A.shape   # Get the shape of input tensor: [n1, n2, ..., nd]
    dim = len(shape)  # Get the number of dimension
    delta = (epsilon / np.sqrt(dim - 1)) * tl.norm(A, 2)  # Truncation parameter

    # Right-to-left orthogonalization
    iterlist = list(range(1, dim))  # Create iteration list: 1, 2, ..., d-1
    iterlist.reverse()              # Reverse the iteration list: d-1, ..., 1 
    for i in iterlist:
        matG = tl.reshape(tensor_train[i], [])
        pass

    # Compression of the orthogonalized representation
    # TODO...    

    return