from utils import Timer, timestamp_msg
import numpy as np
import os
import gzip

import nltk

DATA_DIR = './data'

penntree_tagged_words = None
penntree_tagged_sents = None        

protein_seqs = None
protein_meta = None

def _load_penntreebank():
    
    global penntree_tagged_words, penntree_tagged_sents
    if penntree_tagged_words is None or penntree_tagged_sents is None:
        nltk.download('treebank')
        from nltk.corpus import treebank

        # Organized in sentences and words
        penntree_tagged_sents = treebank.tagged_sents()
        penntree_tagged_words = [word for sent in penntree_tagged_sents for word in sent]
    else:
        timestamp_msg('Using already loaded sequences ...')
    return penntree_tagged_sents, penntree_tagged_words

# Turns sequence of symbols of any hashable object-type into sequence of numerical symbols
# sequence has to be of the form [seq1, seq2, ...] with seq1 = [symb11, symb12, ...]
# type of symbij has to be hashable
def _obj_symb_to_numerical_symb(seqs):
    
    seqs = list(seqs)
    n_seqs = len(seqs)
    
    obj_symbs = list({obj_symb for seq in seqs for obj_symb in seq})
    num_symb_to_obj_symb = {i: obj_symbs[i] for i in range(len(obj_symbs))}
    obj_symb_to_num_symb = {obj_symbs[i]: i for i in range(len(obj_symbs))}
    num_seqs = [np.array([obj_symb_to_num_symb[obj_symb] for obj_symb in seq], dtype=np.int) for seq in seqs]
    
    return num_seqs, num_symb_to_obj_symb, obj_symb_to_num_symb

# Maps symbols of low frequency to a collective rest symbol
# Symbols are of low frequency if the sum of their frequency does not exceed given thresh
# Given sequences need to be numerical
def _map_low_freq_symbs_to_rest_symb(seqs, num_symb_to_obj_symb, obj_symb_to_num_symb, thresh):
    
    symbs, symb_counts = np.unique(np.concatenate(seqs).reshape((-1,)), return_counts=True)
    n_symbs = np.sum(symb_counts)

    symb_sort_idx = np.argsort(symb_counts)
    cumsum_symb_counts = np.cumsum(symb_counts[symb_sort_idx])

    low_freq_symb_idxs = np.where(cumsum_symb_counts < thresh * n_symbs)[0]
    high_freq_symb_idxs = np.where(cumsum_symb_counts >= thresh * n_symbs)[0]
    
    low_freq_symbs = symbs[symb_sort_idx[low_freq_symb_idxs]]
    high_freq_symbs = symbs[symb_sort_idx[high_freq_symb_idxs]]
    
    low_freq_symbol = len(high_freq_symbs)
    
    pruned_num_symb_to_orig_symb = {i: num_symb_to_obj_symb[high_freq_symbs[i]] for i in range(len(high_freq_symbs))}
    pruned_num_symb_to_orig_symb[low_freq_symbol] = [num_symb_to_obj_symb[i] for i in low_freq_symbs]
    
    pruned_orig_symb_to_num_symb = {num_symb_to_obj_symb[high_freq_symbs[i]]: i for i in range(len(high_freq_symbs))}
    pruned_orig_symb_to_num_symb.update({num_symb_to_obj_symb[i] : low_freq_symbol for i in low_freq_symbs})
    
    return pruned_num_symb_to_orig_symb, pruned_orig_symb_to_num_symb, low_freq_symbol
    
def penntreebank_tag_sequences(thresh=0.1, len_cut=None):
    
    t = Timer()
    t.tic("Loading Penntreebank sequences ...")
    
    tagged_sents, tagged_words = _load_penntreebank()
    
    # Cutoff sequences that are too long
    if len_cut is not None:
        tagged_sents = [sent[:len_cut] for sent in tagged_sents]
    
    # Turn into tag-sequence
    tag_seqs = [[tagged_word[1] for tagged_word in tagged_sent] for tagged_sent in tagged_sents]
    sequences, symb_to_tag, tag_to_symb = _obj_symb_to_numerical_symb(tag_seqs)
    
    #seqlens = np.array([len(seq) for seq in sequences])
    #print("#(seqlens > len_cut) / #sequences after cut:", float(len(seqlens[seqlens > len_cut])) / len(sequences))
    """ The symbol frequency is:
    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]),
 array([  766,  1460,  3043,   712,  2134,   824,   694,  1321,   445,
         5834,   241,  2125,  8165,    16,   216,   724, 13166,     4,
         4886,  3874,  2179,  1716,  2822,   120,   563,  6047,  9410,
         2554,   178,  2265,    13,    27,    35,   136,  3546,  6592,
           88,  9857,    14,   927,   244,   126,   182,   381,     1,
            3]))
        Since some appear not often enough, we map multiple symbols of low frequency to one symbol
    """
    
    pruned_symb_to_tag, pruned_tag_to_symb, low_freq_symbol = _map_low_freq_symbs_to_rest_symb(sequences, symb_to_tag, tag_to_symb, thresh)
    
    pruned_sequences = [np.array([ [pruned_tag_to_symb[tag]] for tag in tag_seq]) for tag_seq in tag_seqs]
        
    t.toc("Loaded tag sequences. "
          "Number of words: %d, Number of sentences: %d, Number of symbols: %d" % (len(tagged_words), len(tagged_sents), low_freq_symbol+1))

    return pruned_sequences, pruned_symb_to_tag, pruned_tag_to_symb

def _load_protein():
    
    global protein_seqs, protein_meta
    if protein_seqs is None or protein_meta is None or (len(protein_seqs) != len(protein_meta)):
    
        protein_seqs, protein_meta = [], []
        
        # Parsing data
        filename = os.path.join(DATA_DIR, 'pdb_seqres.txt.gz')
        with gzip.open(filename) as f:
            for i, line in enumerate(f):
                #if i >= 500:
                #    break

                #print(line)
                if line.startswith('>'):
                    protein_meta.append(line.strip())
                else:
                    seq_str = line.strip()
                    protein_seqs.append([c for c in seq_str])
                    
    else:
        timestamp_msg('Using already loaded sequences ...')

    return protein_seqs, protein_meta
    
def protein_sequences(thresh=0.05, len_cut=None, total_perc=1.):
    
    if not (0 < total_perc <= 1):
        raise Exception("Given total_perc has to lie in (0, 1] (Given: %s)" % str(total_perc))
    
    t = Timer()
    t.tic("Loading Protein sequences ...")
    
    protein_seqs, protein_meta = _load_protein()
    
    # Cutoff sequences that are too long
    n_seqs = int(len(protein_seqs) * float(total_perc))
    if len_cut is not None:
        protein_seqs = [seq[:len_cut] for seq in protein_seqs[:n_seqs]]
    
    sequences, symb_to_letter, letter_to_symb = _obj_symb_to_numerical_symb(protein_seqs)
    
    #print(np.unique( np.concatenate(sequences).reshape((-1,)), return_counts=True))
    #print(np.histogram([len(seq) for seq in sequences], bins=20))
    
    """ The symbol frequency is:
    (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24]), array([10466314,  2340962,       33,  7995595,  6777749,  9915067,
        4712172,  6809397,  3213676,  7329805,  2830581, 10935219,
              6,  5114844,  4603616,  5645638,  7662146,  6473431,
         496522,  6902793,  1602712,  8604950,  4147501,   406813,
             62]))
        Since some appear not often enough, we map multiple symbols of low frequency to one symbol
    """
    
    pruned_symb_to_letter, pruned_letter_to_symb, low_freq_symbol = _map_low_freq_symbs_to_rest_symb(sequences, symb_to_letter, letter_to_symb, thresh)
    
    pruned_sequences = [np.array([ [pruned_letter_to_symb[letter]] for letter in protein_seq]) for protein_seq in protein_seqs]
    
    n_symbs = np.sum([len(seq) for seq in pruned_sequences])
    t.toc("Loaded protein sequences. "
          "Number of symbols: %d, Number of sentences: %d, Number of unique symbols: %d" % (n_symbs, len(protein_seqs), low_freq_symbol+1))

    return pruned_sequences, pruned_symb_to_letter, pruned_letter_to_symb
    

def train_test_split(all_sequences, train_perc=0.5, shuffle=True, half=True):
    
    if not (0 < train_perc <= 1):
        raise Exception("Given train_perc has to lie in (0, 1] (Given: %s)" % str(train_perc))
    
    n_sequences = len(all_sequences)
    if half and n_sequences % 2 != 0:
        train_perc = 0.5
        n_sequences -= 1
    
    n_train = int(train_perc * n_sequences)
    n_test = int(n_sequences - n_train)
    
    idx_list = None
    if shuffle:
        idx_list = np.random.choice(n_sequences, size=(n_sequences,), replace=False)
    else:
        idx_list = np.arange(n_sequences)
    
    train_sequences = [np.array(all_sequences[idx_list[i]]) for i in range(n_train)]
    test_sequences = [np.array(all_sequences[idx_list[n_train+i]]) for i in range(n_test)]
    
    #train_symbs = dict()
    #for seq in train_sequences:
    #    sym_seq, count_seq = np.unique(seq, return_counts=True)
    #    for i in range(len(sym_seq)):
    #        sym = sym_seq[i]
    #        if sym in train_symbs.keys():
    #            train_symbs[sym] += count_seq[i]
    #        else:
    #            train_symbs[sym] = 0
    #print(train_symbs)
    
    #test_symbs = dict()
    #for seq in test_sequences:
    #    sym_seq, count_seq = np.unique(seq, return_counts=True)
    #    for i in range(len(sym_seq)):
    #        sym = sym_seq[i]
    #        if sym in test_symbs.keys():
    #            test_symbs[sym] += count_seq[i]
    #        else:
    #            test_symbs[sym] = 0
    #print(test_symbs)
    return train_sequences, test_sequences
    
    