
from models import StandardHMM, DenseHMM, HMMLoggingMonitor
from utils import prepare_data, check_random_state, create_directories, dict_get, Timer, timestamp_msg, check_dir, is_multinomial, compute_stationary, check_sequences
from data import penntreebank_tag_sequences, protein_sequences, train_test_split

from datetime import datetime
import os
import copy

import numpy as np


""" Initializes a StandardHMM and a DenseHMM and fits given data to it
    
"""
def _standard_vs_dense(train_X, test_X, standard_params=None, dense_params=None, gt_AB=None):
    
    t = Timer()
    
    train_X, train_lengths, train_unique = prepare_data(train_X)
    test_X, test_lengths, test_unique = prepare_data(test_X)
    
    standard_hmms = []
    if standard_params is None:
        standard_hmms.append(StandardHMM())
    elif type(standard_params) is list or type(standard_params) is tuple:
        for params in standard_params:
            standard_hmms.append(StandardHMM(**params))
    else:
        standard_params = dict(standard_params)
        standard_hmms.append(StandardHMM(**standard_params))
    
    dense_params = {} if dense_params is None else dict(dense_params)
    dense_hmm = DenseHMM(**dense_params)
    
    opt_schemes = dict_get(dense_params, 'opt_schemes', default=('em', 'cooc'))
    if 'em' in opt_schemes:
        t.tic("Fitting dense HMM in mode 'em' ...")
        dense_hmm.fit(train_X, train_lengths, test_X, test_lengths)
        t.toc("Fitting finished.")
    if 'cooc' in opt_schemes:
        t.tic("Fitting dense HMM in mode 'cooc' ...")
        dense_hmm.fit_coocs(train_X, train_lengths, test_X, test_lengths, gt_AB)
        t.toc("Fitting finished.")
        
    for i, standard_hmm in enumerate(standard_hmms):
        t.tic("Fitting standard hmm %d/%d" % (i+1, len(standard_hmms)))
        standard_hmm.fit(train_X, train_lengths, test_X, test_lengths)
        t.toc("Fitting finished.")
    
def _dirichlet_random_numbers(alpha_size, sample_size, dirichlet_param, random_state):
        return random_state.dirichlet(np.ones(alpha_size) * dirichlet_param, 
                                      size=(sample_size,))
                     
""" Initializes the transition matrices of given hmm to dirichlet distributions.
    Assumes that random_state is an instance of np.RandomState """
def _dirichlet_matrix_initializer(dirichlet_param, n_hidden_states, n_observables, random_state):
    
    pi = 1. / n_hidden_states * np.ones(n_hidden_states)
    A = _dirichlet_random_numbers(n_hidden_states, n_hidden_states, dirichlet_param, random_state)
    B = _dirichlet_random_numbers(n_observables, n_hidden_states, dirichlet_param, random_state)  # Note: This results in an n x m matrix
    
    return pi, A, B
    

def _stationary_matrix_init(n, m, rng, matrix_init_func):
    pi, A, B = matrix_init_func(n, m, rng)
    pi = compute_stationary(A)
    return pi, A, B

def _default_standard_hmm_init():
    return dict(n_hidden_states=1, startprob_prior=1.0, transmat_prior=1.0, 
                random_state=None, em_iter=10, convergence_tol=1e-2, verbose=False)

def _default_dense_hmm_init():
    return dict(n_hidden_states=1, n_observables=None, startprob_prior=1.0, transmat_prior=1.0, 
                random_state=None, em_iter=10, convergence_tol=1e-10, verbose=False,
                params="ste", init_params="ste", logging_monitor=None, mstep_config=None)

def _compute_fair_standard_n(m, n_dense, l_dense):
    pre = - (m - 1)/2
    discriminant = pre**2 + l_dense*(3*n_dense + m + 1)
    if discriminant < 0:
        raise Exception("Complex solution")
        
    n_plus = pre + np.sqrt(discriminant)
    n_minus = pre - np.sqrt(discriminant)
    n = np.max((n_plus, n_minus))
    if n <= 0:
        raise Exception("Only negative solutions")
    
    return int(np.around(n))
    

def _parse_base_parameters(exp_params, path_dict):
    
    path_dict = dict(path_dict)
    exp_params = dict(exp_params)
    exp_params['standard_params'] = dict_get(exp_params, 'standard_params', default=_default_standard_hmm_init(), cast=dict)
    exp_params['dense_params'] = dict_get(exp_params, 'dense_params', default=_default_dense_hmm_init(), cast=dict)
    exp_params['dense_opt_schemes'] = dict_get(exp_params, 'dense_opt_schemes', default=('em',))
    exp_params['compare_to_fair_standard'] = dict_get(exp_params, 'compare_to_fair_standard', default=False)
    
    return exp_params

def _parse_syntheticgt_parameters(exp_params, path_dict):
    
    exp_params = _parse_base_parameters(exp_params, path_dict)
    exp_params['gt_params'] = dict_get(exp_params, 'gt_params', default=_default_standard_hmm_init(), cast=dict)
    exp_params['n_seqs_train'] = dict_get(exp_params, 'n_seqs_train', default=10, cast=int)
    exp_params['seqlen_train'] = dict_get(exp_params, 'seqlen_train', default=10, cast=int)
    exp_params['n_seqs_test'] = dict_get(exp_params, 'n_seqs_test', default=10, cast=int)
    exp_params['seqlen_test'] = dict_get(exp_params, 'seqlen_test', default=10, cast=int)
    exp_params['gt_stationary'] = dict_get(exp_params, 'gt_stationary', default=False)
    exp_params['gt_params']['n_observables'] = exp_params['n_emissions']
    
    # Making sure the initializer returns stationary pi (if gt_stationary = true)...
    init_params = dict_get(exp_params['gt_params'], 'init_params', default=None)
    if init_params is not None and callable(init_params) and exp_params['gt_stationary']:
        init_params_ = lambda n, m, rng: _stationary_matrix_init(n, m, rng, init_params)
        init_params = init_params_
    exp_params['gt_params']['init_params'] = init_params
    
    if 'experiment_directory' in path_dict:
        exp_dir = str(path_dict['experiment_directory'])
        check_dir(exp_dir)
        
        # Log GT EM optimization by default
        gt_log_config = dict_get(exp_params, 'gt_log_config', default=dict(), cast=dict)
        gt_log_config['exp_folder'] = dict_get(gt_log_config, 'exp_folder', default=exp_dir)
        gt_log_config['log_folder'] = dict_get(gt_log_config, 'log_folder', default='/gt_logs/em_opt')
        gt_logmon = HMMLoggingMonitor(gt_log_config)
        exp_params['gt_params']['logging_monitor'] = gt_logmon
        
    return exp_params

def _parse_syntheticgt_dirichlet_parameters(exp_params, path_dict):
    
    exp_params = _parse_syntheticgt_parameters(exp_params, path_dict)
    exp_params['dirichlet_param'] = dict_get(exp_params, 'dirichlet_param', default=0.1, cast=float)
    exp_params['n_emissions'] = dict_get(exp_params, 'n_emissions', default=None)
    
    # Initialize ground truth hmm
    def _dirichlet_matrix_init(n, m, rng):
        return _dirichlet_matrix_initializer(exp_params['dirichlet_param'], n, m, rng)
    
    init_params = dict_get(exp_params['gt_params'], 'init_params', default=_dirichlet_matrix_init)
    if init_params is not None and callable(init_params) and exp_params['gt_stationary']:
        init_params_ = copy.deepcopy(init_params)
        init_params = lambda n, m, rng: _stationary_matrix_init(n, m, rng, init_params_)
    exp_params['gt_params']['init_params'] = init_params
    
    return exp_params

def _parse_standard_and_dense(exp_params, path_dict, n_emissions):
    
    exp_params['n_emissions'] = n_emissions
    
    # Number of emissions must be the same for all models
    exp_params['standard_params']['n_observables'] = n_emissions
    exp_params['dense_params']['n_observables'] = n_emissions
    
    # Set opt_schemes that are needed
    exp_params['dense_params']['opt_schemes'] = exp_params['dense_opt_schemes']
    
    # Setup fair standard hmm
    if exp_params['compare_to_fair_standard']:
        # TODO check l_uz = l_vw
        n_dense, l_dense = exp_params['dense_params']['n_hidden_states'], exp_params['dense_params']['mstep_config']['l_uz'] 
        n_fair = _compute_fair_standard_n(exp_params['n_emissions'], n_dense, l_dense)
        exp_params['fair_standard_params'] = copy.deepcopy(exp_params['standard_params'])
        exp_params['fair_standard_params']['n_hidden_states'] = n_fair
    
    if 'experiment_directory' in path_dict:
        exp_dir = str(path_dict['experiment_directory'])
        check_dir(exp_dir)
        
        standard_log_config = dict_get(exp_params, 'standard_log_config', default=dict(), cast=dict)
        dense_log_config = dict_get(exp_params, 'dense_log_config', default=dict(), cast=dict)
        standard_log_config['exp_folder'] = dict_get(standard_log_config, 'exp_folder', default=exp_dir)
        standard_log_config['log_folder'] = dict_get(standard_log_config, 'log_folder', default='/standard_logs')
        dense_log_config['exp_folder'] = dict_get(dense_log_config, 'exp_folder', default=exp_dir)
        dense_log_config['log_folder'] = dict_get(dense_log_config, 'log_folder', default='/dense_logs')
        
        standard_logmon, dense_logmon = HMMLoggingMonitor(standard_log_config), HMMLoggingMonitor(dense_log_config)
        exp_params['standard_params']['logging_monitor'] = standard_logmon
        exp_params['dense_params']['logging_monitor'] = dense_logmon
        
        fair_standard_logmon = None
        if 'fair_standard_params' in exp_params:
            fair_standard_log_config = dict_get(exp_params, 'fair_standard_log_config', default=dict(), cast=dict)
            fair_standard_log_config['exp_folder'] = dict_get(fair_standard_log_config, 'exp_folder', default=exp_dir)
            fair_standard_log_config['log_folder'] = dict_get(fair_standard_log_config, 'log_folder', default='/fair_standard_logs')
            fair_standard_logmon = HMMLoggingMonitor(fair_standard_log_config)
            exp_params['fair_standard_params']['logging_monitor'] = fair_standard_logmon

    return exp_params
    
    
def _sample_sequences_from_gt_hmm(exp_params, path_dict, gt_hmm=None, sample_retries=100):
    
    t = Timer()
    n_emissions = exp_params['gt_params']['n_observables']
    
    if gt_hmm is None:
        gt_hmm = StandardHMM(**exp_params['gt_params'])
    
    # Sample train and test sequences, save them
    t.tic()
    cur_sample_try = 0
    train_X = None
    while cur_sample_try < sample_retries and not is_multinomial(train_X, min_symbols=n_emissions):
        train_X = gt_hmm.sample_sequences(exp_params['n_seqs_train'], exp_params['seqlen_train'])
        cur_sample_try += 1
    
    if not is_multinomial(train_X, min_symbols=n_emissions):
        raise Exception("Could not sample a multinomial distribution. Try to increase sequence length and number of sequences. Or change the dirichlet parameter")
    
    cur_sample_try = 0
    test_X = None
    while cur_sample_try < sample_retries and not is_multinomial(test_X, min_symbols=n_emissions):
        test_X = gt_hmm.sample_sequences(exp_params['n_seqs_test'], exp_params['seqlen_test'])
        cur_sample_try += 1
    t.toc("Generated train and test sequences")
    
    if not is_multinomial(train_X, min_symbols=n_emissions):
        raise Exception("Could not sample a multinomial distribution. Try to increase sequence length and number of sequences. Or change the dirichlet parameter.")
    
    t.tic()
    if 'gt_dir' in path_dict:
        gt_dir = str(path_dict['gt_dir'])
        check_dir(gt_dir)
        np.save(gt_dir + '/transmat', gt_hmm.transmat_)
        np.save(gt_dir + '/emissionprob', gt_hmm.emissionprob_)
        np.save(gt_dir + '/startprob', gt_hmm.startprob_)
        
        gt_samples = dict_get(exp_params, 'gt_samples', default=None, cast=tuple)
    t.toc("Ground truth parameters logged")
    
    gt_AB = None
    if exp_params['gt_stationary']:
        gt_AB = (gt_hmm.transmat_, gt_hmm.emissionprob_)
        
    _save_data(path_dict, train_X, test_X, gt_AB)
        
    return train_X, test_X, gt_AB

def _save_data(path_dict, train_X, test_X=None, gt_AB=None):
    if 'data_dir' in path_dict:
        data_dir = str(path_dict['data_dir'])
        check_dir(data_dir)
        np.save(data_dir + '/train_X', train_X)
        if test_X is not None:
            np.save(data_dir + '/test_X', test_X)
        if gt_AB is not None:
            np.save(data_dir + '/gt_A', gt_AB[0])
            np.save(data_dir + '/gt_B', gt_AB[1])
        timestamp_msg("Saved data in %s" % data_dir)
    

def _save_experiment_parameters(exp_params, path_dict):
    
    if 'experiment_directory' in path_dict:
        exp_dir = str(path_dict['experiment_directory'])
        check_dir(exp_dir)
    
        _exp_params = copy.deepcopy(exp_params)

        gt_params = dict_get(_exp_params, 'gt_params', default=None, cast=dict)
        if gt_params is not None:
            _exp_params['gt_params'] = gt_params
            init_params = dict_get(gt_params, 'init_params', default=None)
            if callable(init_params):
                _exp_params['gt_params']['init_params'] = str(init_params.__name__)
            gt_logmon = dict_get(gt_params, 'logging_monitor', default=None)
            if gt_logmon is not None and isinstance(gt_logmon, HMMLoggingMonitor):
                _exp_params['gt_params']['logging_monitor'] = dict(gt_logmon.log_config)

        standard_params = dict_get(_exp_params, 'standard_params', default=None, cast=dict)
        standard_logmon = dict_get(standard_params, 'logging_monitor', default=None)
        if standard_logmon is not None and isinstance(standard_logmon, HMMLoggingMonitor):
            _exp_params['standard_params']['logging_monitor'] = dict(standard_logmon.log_config)

        dense_params = dict_get(_exp_params, 'dense_params', default=None, cast=dict)
        dense_logmon = dict_get(standard_params, 'logging_monitor', default=None)
        if dense_logmon is not None and isinstance(dense_logmon, HMMLoggingMonitor):
            _exp_params['dense_params']['logging_monitor'] = dict(dense_logmon.log_config)

        fair_standard_params = dict_get(_exp_params, 'fair_standard_params', default=None, cast=dict)
        fair_standard_logmon = dict_get(fair_standard_params, 'logging_monitor', default=None)
        if fair_standard_logmon is not None and isinstance(fair_standard_logmon, HMMLoggingMonitor):
            _exp_params['fair_standard_params']['logging_monitor'] = dict(fair_standard_logmon.log_config)
            
        np.save(exp_dir + '/exp_params', _exp_params)
        timestamp_msg("Saved experiment parameters in %s" % exp_dir)
        return _exp_params

def synthetic_sequences_experiment(exp_params, path_dict, sample_retries=100, reuse_sequences=None):
    
    t_exp = Timer()
    start_time = t_exp.tic("Starting a 'synthetic sequences' experiment.")
    
    # Get parameters
    t = Timer()
    t.tic("Parsing parameters ...")
    exp_params = _parse_syntheticgt_dirichlet_parameters(exp_params, path_dict)
    exp_params = _parse_standard_and_dense(exp_params, path_dict, exp_params['n_emissions'])
    _exp_params = _save_experiment_parameters(exp_params, path_dict)
    t.toc("Parameters parsed. Using parameters: %s" % str(_exp_params))
    
    train_X, test_X, gt_AB = None, None, None
    if reuse_sequences is None or type(reuse_sequences) != tuple or len(reuse_sequences) != 3:
        train_X, test_X, gt_AB = _sample_sequences_from_gt_hmm(exp_params, path_dict, sample_retries=sample_retries)
    else:
        train_X, test_X, gt_AB = reuse_sequences
        timestamp_msg("Reusing sequences")
    
    if 'fair_standard_params' in exp_params: 
        _standard_vs_dense(train_X, test_X, (exp_params['standard_params'], exp_params['fair_standard_params']),
                           exp_params['dense_params'], gt_AB)
    else:
        _standard_vs_dense(train_X, test_X, exp_params['standard_params'], exp_params['dense_params'], gt_AB)
    
    fin_time, diff = t_exp.toc("Finished a 'synthetic sequences' experiment.")

SUPPORTED_DATASETS = frozenset(('penntree_tag','protein'))
def get_dataset_sequences(ident, ds_params={}, log_dir=None):
    if ident not in SUPPORTED_DATASETS:
        raise Exception("Given Dataset %s is not supported." % str(ident))
    
    sequences, tag_to_symb, symb_to_tag = None, None, None
    if ident == 'penntree_tag':
        sequences, tag_to_symb, symb_to_tag = penntreebank_tag_sequences(**ds_params)
    elif ident == 'protein':
        sequences, tag_to_symb, symb_to_tag = protein_sequences(**ds_params)
    
    if log_dir is not None:
        np.save(log_dir + '/symb_to_tag.npy', symb_to_tag)
        np.save(log_dir + '/tag_to_symb.npy', tag_to_symb)
    
    return sequences, tag_to_symb, symb_to_tag
    

def dataset_synthetic_sequences_experiment(exp_params, path_dict, sample_retries=100):
    
    t_exp = Timer()
    exp_params = dict(exp_params)
    ident = dict_get(exp_params, 'dataset_ident', default='', cast=str)
    start_time = t_exp.tic("Starting a 'dataset synthetic sequences' experiment. (%s)" % str(ident))
    
    gt_dir = dict_get(path_dict, 'gt_dir', default=None)
    check_dir(gt_dir)
    ds_params = dict_get(exp_params, 'dataset_params', default=dict(), cast=dict)
    gt_sequences, _, _ = get_dataset_sequences(ident, ds_params, gt_dir)
    
    # Get parameters
    t = Timer()
    t.tic("Parsing parameters ...")
    
    # Check gt_sequences
    sequences, lengths, n_emissions = check_sequences(gt_sequences)
    exp_params['n_emissions'] = n_emissions
    
    exp_params = _parse_syntheticgt_parameters(exp_params, path_dict)
    exp_params = _parse_standard_and_dense(exp_params, path_dict, exp_params['n_emissions']) 
    _exp_params = _save_experiment_parameters(exp_params, path_dict)
    t.toc("Parameters parsed. Using parameters: %s" % str(_exp_params))
    
    t.tic("Fitting GT HMM...")
    gt_hmm = StandardHMM(**exp_params['gt_params'])
    gt_hmm.fit(sequences, lengths)
    t.toc("Fitting finished")
    
    train_X, test_X, gt_AB = _sample_sequences_from_gt_hmm(exp_params, path_dict, gt_hmm=gt_hmm, sample_retries=sample_retries)
    
    if 'fair_standard_params' in exp_params: 
        _standard_vs_dense(train_X, test_X, (exp_params['standard_params'], exp_params['fair_standard_params']),
                           exp_params['dense_params'], gt_AB)
    else:
        _standard_vs_dense(train_X, test_X, exp_params['standard_params'], exp_params['dense_params'], gt_AB)
    
    fin_time, diff = t_exp.toc("Finished a 'dataset synthetic sequences' experiment.")
    
def dataset_sequences_experiment(exp_params, path_dict, reuse_sequences=None):
    
    t_exp = Timer()
    exp_params = dict(exp_params)
    ident = dict_get(exp_params, 'dataset_ident', default='', cast=str)
    start_time = t_exp.tic("Starting a 'dataset sequences' experiment. (%s)" % str(ident))
    
    # Get parameters
    t = Timer()
    t.tic("Parsing parameters ...")
    
    train_perc = dict_get(exp_params, 'train_perc', default=1., cast=float)
    gt_dir = dict_get(path_dict, 'gt_dir', default=None)
    check_dir(gt_dir)
    ds_params = dict_get(exp_params, 'dataset_params', default=dict(), cast=dict)
    if reuse_sequences is None or type(reuse_sequences) != tuple or len(reuse_sequences) != 2:
        gt_sequences, _, _ = get_dataset_sequences(ident, ds_params, gt_dir)
        train_X, test_X = train_test_split(gt_sequences, train_perc)
    else:
        train_X, test_X = reuse_sequences
        timestamp_msg("Reusing sequences ...")
    
    # Check gt_sequences
    _, _, n_train_emissions = check_sequences(train_X)
    n_test_emissions = None
    if test_X is not None and len(test_X) > 0:
        _, _, n_test_emissions = check_sequences(test_X)
    _save_data(path_dict, train_X, test_X)
    if n_test_emissions is not None and n_train_emissions != n_test_emissions:
        raise Exception("Number of emissions in train and test sequence differs")
    exp_params['n_emissions'] = n_train_emissions
    
    exp_params = _parse_base_parameters(exp_params, path_dict)
    exp_params = _parse_standard_and_dense(exp_params, path_dict, exp_params['n_emissions']) 
    _exp_params = _save_experiment_parameters(exp_params, path_dict)
    t.toc("Parameters parsed. Using parameters: %s" % str(_exp_params))
    
    if 'fair_standard_params' in exp_params: 
        _standard_vs_dense(train_X, test_X, (exp_params['standard_params'], exp_params['fair_standard_params']),
                           exp_params['dense_params'])
    else:
        _standard_vs_dense(train_X, test_X, exp_params['standard_params'], exp_params['dense_params'])
    
    fin_time, diff = t_exp.toc("Finished a 'dataset sequences' experiment.")

def run_experiment(exp_type, exp_name, exp_params, reuse_setup=None):
    
    experiment_directory, path_dict = None, None
    if reuse_setup is None or type(reuse_setup) != tuple or len(reuse_setup) != 2:
        experiment_directory, path_dict = setup_experiment(exp_name, exp_params)
    else:
        experiment_directory, path_dict = reuse_setup
        
    supported_exp_types = ('synthetic_sequences', 'dataset_synthetic_sequences', 'dataset_sequences')
    if exp_type == 'synthetic_sequences':
        synthetic_sequences_experiment(exp_params, path_dict)
        
    elif exp_type == 'dataset_synthetic_sequences':
        dataset_synthetic_sequences_experiment(exp_params, path_dict)
    
    elif exp_type == 'dataset_sequences':
        dataset_sequences_experiment(exp_params, path_dict)
        
    else:
        raise Exception('Given experiment type "%s" is not supported. \n'
                        'It has to be one of the following: %s' % (str(exp_type), str(supported_exp_types)))
        
    print(experiment_directory)
    return experiment_directory

def setup_experiment(exp_name, exp_params):
    path_dict = {}
    experiment_directory = os.getcwd() + '/' + exp_name + datetime.now().strftime('%Y%m%d_%H-%M-%S')
    path_dict['experiment_directory'] = experiment_directory
    path_dict['data_dir'] = experiment_directory + '/data'
    path_dict['gt_dir'] = experiment_directory + '/gt_logs'
    return experiment_directory, path_dict
    
    