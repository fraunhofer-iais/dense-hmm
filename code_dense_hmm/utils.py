from sklearn.utils import check_array, check_random_state
import numpy as np
import os
import datetime as dt

# Following is used in experiment.py

""" Checks if X is a list of arrays of 2 dimensions
    Turns X into an array of 2 dimensions and computes an array (lengths) containing
    the lengths of the individual sequences
    If n_emissions is given, it also checks whether the data contains n_emissions many
    unique symbols
    Returns the prepared X, lengths and a set of unique symbols in X
"""
def prepare_data(X, n_emissions=None):
        
    seqs = [np.array(seq) for seq in X]
    if any(len(seq.shape) != 2 for seq in seqs):
        raise Exception("Given data has to be a list of arrays of 2 dimensions.")
    lengths = np.array([seq.shape[0] for seq in seqs])

    X = np.concatenate(seqs)
    unique_symbs = {seq_elem[0] for seq_elem in X}

    if n_emissions is not None:
        n_unique_symbs = len(unique_symbs)
        if n_unique_symbs != n_emissions:
            print("n_emissions parameter set to %d, "
                  "but data doesnt have that many emissions. (Has %d)" % 
                  (n_emissions, n_unique_symbs))

    return X, lengths, unique_symbs
                  
def dict_get(d, key, default=None, cast=None):
    entry = d[key] if d is not None and key in d and d[key] is not None else default
    if entry is not None and cast is not None and callable(cast):
        return cast(entry)
    return entry
                 
""" Creates directories using given list dirs """
def create_directories(dirs):
    for dir in dirs:
        os.makedir(dir)

""" Timestamp of the format: hour:minute:second.microsecond """                  
def timestamp(dt_obj):
    return "%d:%d:%d.%.6d:" % (dt_obj.hour, dt_obj.minute, 
                               dt_obj.second, dt_obj.microsecond)

def timestamp_msg(msg):
    dt_now = dt.datetime.now()
    tstamp = timestamp(dt_now)
    if msg is not None:
        print("%s - %s" % (tstamp, str(msg)))
    return tstamp, dt_now
                  
""" Timer class providing a function tic() that returns the current time and toc() that
returns the current time and the time difference since tic() was called last time """
class Timer:
    def __init__(self):
        self.last_datetime = None
        self.tic()
                  
    def tic(self, msg=None):
        tstamp, dt_now = timestamp_msg(msg)
        self.last_datetime = dt_now
        return tstamp
                  
    def toc(self, msg=None):
        td = str(dt.datetime.now() - self.last_datetime)
        tstamp = timestamp(self.last_datetime)
        
        if msg is not None:
            print("%s - %s (took %s)" % (tstamp, msg, td))
        return tstamp, td
                  
def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def is_multinomial(X, min_symbols=None):
    symbols = np.reshape(X, -1)
    if (len(symbols) == 1                                # not enough data
        or not np.issubdtype(symbols.dtype, np.integer)  # not an integer
        or (symbols < 0).any()):                         # not positive
        return False
    u = np.unique(symbols)
    
    if min_symbols is not None and len(u) < min_symbols:
        return False
    
    return u[0] == 0 and u[-1] == len(u) - 1

def check_sequences(sequences):
    
    # Either of form [w1, w2, ..., wn]
    gt_lengths = None
    if np.all([np.isscalar(w) for w in sequences]):
        sequences = np.array(sequences)[:, np.newaxis]
        lengths = np.array([sequences.shape[0]])
    # Or [[w11, w12, ...], [w21, w22, ...]]
    elif np.all([np.isscalar(w) for seq in sequences for w in seq]):
        lengths = np.array([len(seq) for seq in sequences])
        sequences = np.concatenate(sequences)[:, np.newaxis]
    # Or [[[w11], [w12], ...], [[w21], [w22], ...]]
    elif np.all([len(w) == 1 for seq in sequences for w in seq]):
        lengths = np.array([len(seq) for seq in sequences])
        sequences = np.concatenate(sequences)
    else:
        raise Exception("Given sequences have unsupported shape.")
    
    if not np.issubdtype(sequences.dtype, np.integer):
            timestamp_msg("Warning: Given sequences are not integer. Casting them to int ...")
    sequences.astype(np.integer)
    n_emissions = len(np.unique(sequences.reshape(-1)))
    
    if not is_multinomial(sequences):
        raise Exception("Given sequences are not a sample from a Multinomial distribution.")
        
    return sequences, lengths, n_emissions


# Following is used in models.py
                      
""" Checks if X is an array and has a shape of (?, 1). 
    Computes the number of sequences n_seqs in X and the maximal sequence length max_seqlen
    Returns X, n_seqs, max_seqlen
"""
def check_arr(X, lengths=None):
    
    X = check_array(X)
    
    if X.shape[-1] != 1:
        raise Exception("Sequences of vectors are currently not supported. Please map each vector to an integer value")
    
    # A sequence in the form of [[o1],[o2],...] has to be given
    if len(X.shape) != 2:
        raise Exception("Sequence array X must be of dimension 2")
    
    n_seqs = 1
    if lengths is None:
        max_seqlen = X.shape[0]
    else:
        if X.shape[0] != np.sum(lengths):
            raise Exception("Number of elements in given sequence matrix does not match given lengths array")
        n_seqs = len(lengths)
        max_seqlen = max(lengths) 
    
    return X, n_seqs, max_seqlen

""" Pads first dimension of X to the right: returns pad_X, such that pad_X.shape = (seqlen, pad_X.shape[1]) """
def pad_to_seqlen(X, seqlen, mode='constant', constant_values=(0), **kwargs):
    
    X = check_array(X)
    
    diff_to_seqlen = seqlen - X.shape[0]
    
    pad_width = list((0, 0) for _ in X.shape)
    pad_width[0] = (0, diff_to_seqlen) # Pad first dimension to the right
    
    pad_X = np.pad(X, pad_width, mode=mode, constant_values=constant_values, **kwargs)
    return pad_X

def compute_stationary(M, verbose=True):
    
    eigval, eigvec = np.linalg.eig(M.T)
    idx = np.asarray(np.isclose(eigval, [1.])).nonzero()[0]
    if idx.size < 1:
        raise Exception("No Eigenvalue 1")
    elif idx.size > 1 and verbose:
        print("Warning: Multiple vectors corresponding to eigenvalue 1.: %s" % str(idx))
    M_stationary = eigvec[:, idx[0]].real
    M_stationary = M_stationary / np.sum(M_stationary)
    return M_stationary   

# Following is used in plots.py

# loads npy/npz file if it exists
def load_if_exists(file, message=None, retval=None, allow_pickle=True):
    
    if os.path.isfile(file):
        return np.load(file, allow_pickle=allow_pickle)
    elif message is not None:
        print(message)
    return retval

def count_none(l):
    return sum([1 for entry in l if entry is None])

# This function expects the matrices in hmmlearn's convention
def analytical_coocs(A, B, pi):
    
    # Check pi being stationary
    if(not np.all(np.isclose(np.matmul(A.T, pi), pi))):
        print("Warning: Given pi is not stationary! Ignoring pi and computing the stationary of A instead!")
        pi = compute_stationary(A)
    
    theta = A * pi[:, None]
    omega = np.matmul(B.T, np.matmul(theta, B))
    return omega

def _lengths_iterator(seqs, lengths):
    n_seqs = len(lengths)
    left, right = 0, 0
    
    for i in range(len(lengths)):
        right += lengths[i]
        yield seqs[left:right]
        left += lengths[i]
        
# If lengths=None, seqs is assumed to be of the form [seq1, seq2, ...]
# If lengths!=None, seqs is assumed to be of the form [symb1, symb2, ...]
def empirical_coocs(seqs, m, lengths=None):
        
    freqs = np.zeros((m, m))
    seq_iterator = seqs
    if lengths is not None:
        seq_iterator = _lengths_iterator(seqs, lengths)
    
    for seq in seq_iterator:

        if seq.shape[0] <= 1: # no transitions
            continue

        seq = seq.reshape(-1)

        seq_pairs = np.dstack((seq[:-1], seq[1:]))
        seq_pairs, counts = np.unique(seq_pairs, return_counts=True, axis=1)
        seq_pre, seq_suc = [arr.flatten() for arr in np.dsplit(seq_pairs, 2)]
        freqs[seq_pre, seq_suc] += counts

    return freqs, np.reshape(freqs / np.sum(freqs), newshape=(-1))

def cooc_loss(x, y):
    return np.mean(np.abs(y-x))
    
    
    
    