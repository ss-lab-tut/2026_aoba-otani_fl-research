"""Index partitions which neither discard remainders nor duplicate samples."""
import numpy as np


def client_indices(n_samples,client_id,num_clients):
    if not all(isinstance(v,(int,np.integer)) for v in (n_samples,client_id,num_clients)):
        raise ValueError('integer counts and client id required')
    if num_clients<=0 or n_samples<num_clients or not 0<=client_id<num_clients:
        raise ValueError('require nonempty clients and valid client id')
    return np.array_split(np.arange(n_samples),num_clients)[client_id]
