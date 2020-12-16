import matplotlib.pyplot as plt, numpy as np, os
from utils import load_if_exists, count_none, dict_get, analytical_coocs, empirical_coocs, cooc_loss, Timer

SMALL_SIZE = 15
MEDIUM_SIZE = 40
BIGGER_SIZE = 45

plt.rc('font', size=BIGGER_SIZE)# controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=BIGGER_SIZE, linewidth=5)     # fontsize of the axes title # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('xtick.major', width=5, size=20)
plt.rc('xtick.minor', width=5, size=20)
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick.major', width=5, size=20)
plt.rc('ytick.minor', width=5, size=20)
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('lines', linewidth=5)


def _n_seqs_seqlen(exp_params, default_n_seqs_train=None, default_seqlen_train=None, default_n_seqs_test=None, default_seqlen_test=None):
    
    n_seqs_train = dict_get(exp_params, 'n_seqs_train', default=default_n_seqs_train, cast=int)
    seqlen_train = dict_get(exp_params, 'seqlen_train', default=default_seqlen_train, cast=int)
    
    n_seqs_test = dict_get(exp_params, 'n_seqs_test', default=default_n_seqs_test, cast=int)
    seqlen_test = dict_get(exp_params, 'seqlen_test', default=default_seqlen_test, cast=int)
    
    n_seqs = [_ for _ in (n_seqs_train, n_seqs_test) if _ is not None]
    if len(n_seqs) > 0 and np.all(np.array(n_seqs) == n_seqs_train):
        n_seqs = n_seqs_train
        
    seqlen = [_ for _ in (seqlen_train, seqlen_test) if _ is not None]
    if len(seqlen) > 0 and np.all(np.array(seqlen) == seqlen_train):
        seqlen = seqlen_train
        
    return n_seqs, seqlen, n_seqs_train, seqlen_train, n_seqs_test, seqlen_test

def _l(exp_params):
    
    l = None
    dense_params = dict_get(exp_params, 'dense_params', default=None, cast=dict)
    mstep_config = dict_get(dense_params, 'mstep_config', default=None, cast=dict)
    if mstep_config is not None:
        l_uz = dict_get(mstep_config, 'l_uz', default=None, cast=int)
        l_vw = dict_get(mstep_config, 'l_vw', default=None, cast=int)
        
        l = np.array([l_ for l_ in (l_uz, l_vw) if l_ is not None])
        if len(l) == 0:
            l = None
        elif np.all(l == l[0]):
            l = l[0]
            
    return l, l_uz, l_vw
    
def _n_hidden(exp_params):
    
    gt_params = dict_get(exp_params, 'gt_params', default=None, cast=dict)
    standard_params = dict_get(exp_params, 'standard_params', default=None, cast=dict)
    dense_params = dict_get(exp_params, 'dense_params', default=None, cast=dict)
    
    n_standard = dict_get(standard_params, 'n_hidden_states', default=None, cast=int)
    n_dense = dict_get(dense_params, 'n_hidden_states', default=None, cast=int)
    n_gt = dict_get(gt_params, 'n_hidden_states', default=None, cast=int)
    n_hidden = np.array([n_gt, n_standard, n_gt])
    n_hidden = n_hidden[n_hidden != None]
    if np.all(n_hidden[0] == n_hidden):
        n_hidden = n_hidden[0]
    return n_hidden, n_gt, n_standard, n_dense


def _n_em(exp_params):
    
    standard_params = dict_get(exp_params, 'standard_params', default=None, cast=dict)
    dense_params = dict_get(exp_params, 'dense_params', default=None, cast=dict)
    n_em_standard = dict_get(standard_params, 'em_iter', default=None, cast=int)
    n_em_dense = dict_get(dense_params, 'em_iter', default=None, cast=int)
    
    n_em = np.array([n_em_ for n_em_ in (n_em_standard, n_em_dense) if n_em_ is not None])
    if len(n_em) == 0:
        n_em = None
    elif np.all(n_em == n_em[0]):
        n_em = n_em[0]
    
    return n_em, n_em_standard, n_em_dense


def em_file(exp_dir, model, em, ident):
    return '%s/%s_logs/logs_em=%d_ident=%s.npz' % (exp_dir, model, em, ident)

def cooc_file(exp_dir, model):
    return '%s/%s_logs/logs_coocs.npz' % (exp_dir, model)

# Loads npz archive holding logs
# Appends data to a corresponding list in d
def load_if_exists_and_append(d, file, default=None, verbose=False):
    
    message = 'Warning: File %s does not exist' % file if verbose else None
    data = load_if_exists(file, message=message, retval=default, allow_pickle=False)
    
    if data == default:
        if verbose:
            print("Warning: Skipping file %s (does not exist)" % str(file))
        return
    
    for key in d.keys():
        if key in data.keys() and data[key] is not None:
            if type(d[key]) == list:
                d[key].append(data[key])
            else:
                d[key] = data[key]
                
            
def load_exp_params(exp_dir):
    res = np.load(exp_dir + '/exp_params.npy', allow_pickle=True).item()
    return res


label_dict = {'standard': ['EM_loss', 'EM_loss_pi', 'EM_loss_A', 'EM_loss_B'], 'dense': ['EM_loss', 'EM_loss_pi', 'EM_loss_pi_norm', 'EM_loss_A', 'EM_loss_A_norm', 'EM_loss_B', 'EM_loss_B_norm']}
model_label = {'dense': 'DenseHMM', 'standard': 'Standard HMM'}
def plot_loss_and_loglike(exp_params, model, verbose=False):
    
    t = Timer()
    
    exp_params = dict(exp_params)
    labels = label_dict[model]
    params_key = '%s_params' % model
    if params_key not in exp_params:
        raise Exception('Given exp_params dictionary does not provide an entry for the key: %s' % params_key)
    if 'logging_monitor' not in exp_params[params_key]:
        raise Exception("No logging monitor in given exp_params['%s']" % params_key)
    elif 'exp_folder' not in exp_params[params_key]['logging_monitor']:
        raise Exception("Cannot retrieve experiment directory from exp_params['%s']['logging_monitor']['exp_folder']" % params_key)
    elif 'em_iter' not in exp_params[params_key]:
        raise Exception("Key 'em_iter' not in exp_params['%s']" % params_key)

    t.tic()
    data_em = {'loglike': [], 'val_loglike': [], 'train_losses': [], 'test_losses': [], 'test_gamma_losses': []}
    exp_dir = exp_params[params_key]['logging_monitor']['exp_folder']
    n_em = exp_params[params_key]['em_iter']
    for em in range(n_em):
        load_if_exists_and_append(data_em, em_file(exp_dir, model, em, 'aE'), verbose=verbose)
        if em < n_em - 1:
            load_if_exists_and_append(data_em, em_file(exp_dir, model, em, 'aM'), verbose=verbose)
        if em == n_em - 1:
            load_if_exists_and_append(data_em, em_file(exp_dir, model, em, 'f'), verbose=verbose)
    t.toc("Successfully loaded EM files")
    
    data_coocs, data_fair_standard = None, None
    if model == 'dense':
        
        t.tic()
        data_coocs = {'cooc_logprobs': None, 'cooc_val_logprobs': None}
        load_if_exists_and_append(data_coocs, cooc_file(exp_dir, 'dense'))
        
        if 'compare_to_fair_standard' in exp_params and exp_params['compare_to_fair_standard']:
            
            data_fair_standard = {'train_losses': [], 'loglike': []}
            for em in range(n_em):
                load_if_exists_and_append(data_fair_standard, em_file(exp_dir, 'fair_standard', em, 'aE'))
                if em < n_em - 1:
                     load_if_exists_and_append(data_fair_standard, em_file(exp_dir, 'fair_standard', em, 'aM'))
                elif em == n_em - 1:
                     load_if_exists_and_append(data_fair_standard, em_file(exp_dir, 'fair_standard', em, 'f'))
        t.toc("Loaded Cooc files / Fair EM files")
    
    t.tic()
    if len(data_em['loglike']) > 0:
        loglikes = -np.array(data_em['loglike'])
        print("Final loglikelihood: %.5f" % float(loglikes[-1]))
        if len(loglikes.shape) != 1:
            raise Exception('Given loglikes have unsupported shape: %s' % str(loglikes.shape))
        plt.plot(loglikes, label='loglike', marker='x', markersize=2.)
    
    if len(data_em['train_losses']) > 0:
        train_losses = np.array(data_em['train_losses'])
        if len(train_losses.shape) == 1:
            plt.plot(train_losses, label='loss', marker='x', markersize=2.)
        elif len(train_losses.shape) != 2:
            raise Exception('Given train losses have unsupported shape: %s' % str(train_losses.shape))
        else:
            for i in range(train_losses.shape[-1]):
                plt.plot(train_losses[:, i], label=labels[i], alpha=0.5)
            
        if data_coocs is not None and data_coocs['cooc_logprobs'] is not None:
            cooc_loglike = -np.sum(data_coocs['cooc_logprobs'])
            print("Final Cooc likelihood: %.5f" % float(cooc_loglike))
            plt.plot([0, train_losses.shape[0]], [cooc_loglike, cooc_loglike], '--', label='loglike (opt=cooc)') 

            
        if data_fair_standard is not None:
            if data_fair_standard['loglike'] is not None:
                fair_standard_loglikes = -np.array(data_fair_standard['loglike'])
                print("Final loglikelihood (Fair standard): %.5f" % float(fair_standard_loglikes[-1]))
                #fair_standard_loglikes = -fair_standard_loglikes[fair_standard_loglikes != None]
                plt.plot(fair_standard_loglikes, label='Fair StandardHMM loglike')
        
            if data_fair_standard['train_losses'] is not None:
                fair_standard_train_losses = np.array(data_fair_standard['train_losses'])
                if len(fair_standard_train_losses.shape) == 1:
                    plt.plot(fair_standard_train_losses, label='Fair StandardHMM EM-loss')
                elif len(fair_standard_train_losses.shape) != 2:
                    raise Exception('Given train losses (standard fair) have unsupported shape: %s' % str(fair_standard_train_losses.shape))
                else:
                     plt.plot(fair_standard_train_losses[:, 0], label='Fair StandardHMM EM-loss')
           
        t.toc("Plotted data")
    #print('Data successfully read for model %s (n_em: %d). None entries: loglikes: %d, val_loglikes: %d, train_losses %d, test_losses: %d, test_gamma_losses: %d ' % (model, n_em, count_none(loglikes), count_none(val_loglikes), count_none(train_losses), count_none(test_losses), count_none(test_gamma_losses)))
    
    t.tic()
    l, l_uz, l_vw = _l(exp_params)
    title_str = 'train(stationary)' if 'gt_stationary' in exp_params and exp_params['gt_stationary'] else 'train'
    title_str += ', ' + model_label[model] + '\n'
    title_str += ('n_seqs=%d' % exp_params['n_seqs_train']) if 'n_seqs_train' in exp_params else ''
    title_str += (', seqlen=%d' % exp_params['seqlen_train']) if 'n_seqs_train' in exp_params else ''
    title_str += (', m=%d' % exp_params['n_emissions']) if 'n_seqs_train' in exp_params else ''
    title_str += (', n=%s' % exp_params[params_key]['n_hidden_states']) if 'n_hidden_states' in exp_params[params_key] else ''
    title_str += (', l=%s' % str(l)) if l is not None else '' 
    title_str += ', EM_iter=%d' % n_em
    plt.title(title_str)
    plt.ylabel('')
    plt.xlabel('EM iterations')
    plt.legend()
    t.toc("Read params for figure title.")
    
    fig = plt.gcf()
    fig.set_size_inches(14, 6)
    
    plt.show()


""" Plots upto 6 plots: 
- train, test vs cooc/em  (cooc is always the learned omega for now)
- synth vs cooc/em (if an is True; synth are the gt analytical coocs (assumes stationarity))
- train, test vs standard/dense
- synth vs standard/dense
TODO: train, test vs standard/cooc/em?
"""
def plot_coocurrences(exp_params, verbose=False, alpha=0.5, an=False):
    
    t = Timer()
    
    exp_params = dict(exp_params)
    exp_dir = exp_params['standard_params']['logging_monitor']['exp_folder']
    n_obs = dict_get(exp_params, 'n_emissions', default=None, cast=int)
    if n_obs is None:
        raise Exception("'n_emissions' missing in exp_params dict")
        
    gt_params = dict_get(exp_params, 'gt_params', default=None, cast=dict)
    #standard_params = dict_get(exp_params, 'standard_params', default=None, cast=dict)
    #dense_params = dict_get(exp_params, 'dense_params', default=None, cast=dict)
    
    n_em, n_em_standard, n_em_dense = _n_em(exp_params)
    if n_em is None:
        if verbose:
            print('Skipping EM plots as both n_em_standard and n_em_dense is None')
    if hasattr(n_em, '__len__') and len(n_em) > 1:
        n_em = min(n_em_standard, n_em_dense)
        if verbose:
            print('Warning: n_em differs for standard and dense model. Using: %d' % n_em)
            
    """ Read files and compute coocs """
    train_coocs, test_coocs, synth_coocs, standard_coocs, dense_em_coocs, dense_cooc_coocs, standard_fair_coocs, (train_X_n_seqs, train_X_seqlen, test_X_n_seqs, test_X_seqlen) = _compute_coocs(exp_dir, n_obs, n_em, an=an, verbose=verbose)
    
    """ Parse further params """
    
    t.tic()
    
    gt_stationary = dict_get(exp_params, 'gt_stationary', default=None, cast=bool)
    
    n_seqs, seqlen, n_seqs_train, seqlen_train, n_seqs_test, seqlen_test = _n_seqs_seqlen(exp_params, train_X_n_seqs, train_X_seqlen, test_X_n_seqs, test_X_seqlen)
    
    seq_nml_str = ''
    if gt_stationary is not None and gt_stationary:
        seq_nml_str = 'stationary_gt, n_seqs=%s' % str(n_seqs)
    else:
        seq_nml_str = 'n_seqs=%s' % str(n_seqs) 
    seq_nml_str += ', seqlen=%s' % str(seqlen) 
    seq_nml_str += ', m=%d' % n_obs
        
    n_hidden, n_gt, n_standard, n_dense = _n_hidden(exp_params)
    seq_nml_str += (', n=%s' % str(n_hidden)) if n_hidden is not None else ''
    
    l, l_uz, l_vw = _l(exp_params)
    seq_nml_str += (', l=%s' % str(l)) if l is not None else ''
        
    fair_standard_params = dict_get(exp_params, 'fair_standard_params', default=None, cast=dict)
    n_fair_standard_hidden = dict_get(fair_standard_params, 'n_hidden_states', default=None, cast=int)
    n_fair_standard_em = dict_get(fair_standard_params, 'em_iter', default=None, cast=int)
    
    t.tic("Parsed experiment data")
    
    _plot_cooccurences(train_coocs, test_coocs, synth_coocs, standard_coocs, dense_em_coocs, dense_cooc_coocs, standard_fair_coocs, seq_nml_str)

    
def _plot_cooccurences(train_coocs, test_coocs, synth_coocs, standard_coocs, dense_em_coocs, dense_cooc_coocs, standard_fair_coocs, seq_nml_str):
    
    """ Plot """
    
    t.tic("Plotting ...")
    
    def cooc_plot_skeleton(title_str=''):
        plt.plot([1e-10, 1.5], [1e-10, 1.5])
        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.ylim([1e-10, 1.5])
        
        plt.xlabel('GT cooc')
        plt.ylabel('HMM/DenseHMM cooc')
        plt.title(seq_nml_str + title_str)
        plt.legend()
        fig = plt.gcf()
        fig.set_size_inches(14, 6)
        plt.show()
    
    # 1. train vs cooc/em  (cooc is always the learned omega for now)
    filled = False
    if train_coocs is not None:
        if dense_cooc_coocs is not None:
            loss = cooc_loss(train_coocs, dense_cooc_coocs)
            plt.scatter(train_coocs, dense_cooc_coocs, label='gt(empirical, train) vs dense(opt=cooc) [mad=%.4f]' % loss, alpha=alpha)
            filled = True
        if dense_em_coocs is not None:
            loss = cooc_loss(train_coocs, dense_em_coocs[-1, ...])
            plt.scatter(train_coocs, dense_em_coocs[-1, ...], label='gt(empirical, train) vs dense(opt=EM) [mad=%.4f]' % loss, alpha=alpha)
            filled = True
            
        if standard_fair_coocs is not None:
            loss = cooc_loss(train_coocs, standard_fair_coocs[-1, ...])
            plt.scatter(train_coocs, standard_fair_coocs[-1, ...], label='gt(empirical, train) vs standard fair [mad=%.4f]' % loss, alpha=alpha)
            filled = True
    
    if filled:
        if n_fair_standard_hidden is not None:
            cooc_plot_skeleton(', n_fair=%d' % n_fair_standard_hidden)
        else:
            cooc_plot_skeleton()
    
    # 2. test vs cooc/em
    filled = False
    if test_coocs is not None:
        if dense_cooc_coocs is not None:
            loss = cooc_loss(test_coocs, dense_cooc_coocs)
            plt.scatter(test_coocs, dense_cooc_coocs, label='gt(empirical, test) vs dense(opt=cooc) [mad=%.4f]' % loss, alpha=alpha)
            filled = True
        if dense_em_coocs is not None:
            loss = cooc_loss(test_coocs, dense_em_coocs[-1, ...])
            plt.scatter(test_coocs, dense_em_coocs[-1, ...], label='gt(empirical, test) vs dense(opt=EM) [mad=%.4f]' % loss, alpha=alpha)
            filled = True
            
        if standard_fair_coocs is not None:
            loss = cooc_loss(test_coocs, standard_fair_coocs[-1, ...])
            plt.scatter(test_coocs, standard_fair_coocs[-1, ...], label='gt(empirical, test) vs standard fair [mad=%.4f]' % loss, alpha=alpha)
            filled = True
    
    if filled:
        if n_fair_standard_hidden is not None:
            cooc_plot_skeleton(', n_fair=%d' % n_fair_standard_hidden)
        else:
            cooc_plot_skeleton()
        
    # 3. synth vs cooc/em
    filled = False
    if synth_coocs is not None:
        if dense_cooc_coocs is not None:
            loss = cooc_loss(synth_coocs, dense_cooc_coocs)
            plt.scatter(synth_coocs, dense_cooc_coocs, label='gt(analytical) vs dense(opt=cooc) [mad=%.4f]' % loss, alpha=alpha)
            filled = True
        if dense_em_coocs is not None:
            loss = cooc_loss(synth_coocs, dense_em_coocs[-1, ...])
            plt.scatter(synth_coocs, dense_em_coocs[-1, ...], label='gt(analytical) vs dense(opt=EM) [mad=%.4f]' % loss, alpha=alpha)
            filled = True
        if standard_fair_coocs is not None:
            loss = cooc_loss(synth_coocs, standard_fair_coocs[-1, ...])
            plt.scatter(synth_coocs, standard_fair_coocs[-1, ...], label='gt(analytical) vs standard fair [mad=%.4f]' % loss, alpha=alpha)
            filled = True

    if filled:
        if n_fair_standard_hidden is not None:
            cooc_plot_skeleton(', n_fair=%d' % n_fair_standard_hidden)
        else:
            cooc_plot_skeleton()
        
    # 4. train vs standard/dense
    filled = False
    if train_coocs is not None:
        if standard_coocs is not None:
            loss = cooc_loss(train_coocs, standard_coocs[-1, ...])
            plt.scatter(train_coocs, standard_coocs[-1, ...], label='gt(empirical, train) vs standard [mad=%.4f]' % loss, alpha=alpha)
            filled = True
        if dense_em_coocs is not None:
            loss = cooc_loss(train_coocs, dense_em_coocs[-1, ...])
            plt.scatter(train_coocs, dense_em_coocs[-1, ...], label='gt(empirical, train) vs dense(opt=EM) [mad=%.4f]' % loss, alpha=alpha)
            filled = True
        if standard_fair_coocs is not None:
                loss = cooc_loss(train_coocs, standard_fair_coocs[-1, ...])
                plt.scatter(train_coocs, standard_fair_coocs[-1, ...], label='gt(empirical, train) vs standard fair [mad=%.4f]' % loss, alpha=alpha)
                filled = True
            
    if filled:
        if n_fair_standard_hidden is not None:
            cooc_plot_skeleton(', n_fair=%d' % n_fair_standard_hidden)
        else:
            cooc_plot_skeleton()
        
    # 5. test vs standard/dense
    filled = False
    if test_coocs is not None:
        if standard_coocs is not None:
            loss = cooc_loss(test_coocs, standard_coocs[-1, ...])
            plt.scatter(test_coocs, standard_coocs[-1, ...], label='gt(empirical, test) vs standard [mad=%.4f]' % loss, alpha=alpha)
            filled = True
        if dense_em_coocs is not None:
            loss = cooc_loss(test_coocs, dense_em_coocs[-1, ...])
            plt.scatter(test_coocs, dense_em_coocs[-1, ...], label='gt(empirical, test) vs dense(opt=EM) [mad=%.4f]' % loss, alpha=alpha)
            filled = True
        if standard_fair_coocs is not None:
            loss = cooc_loss(test_coocs, standard_fair_coocs[-1, ...])
            plt.scatter(test_coocs, standard_fair_coocs[-1, ...], label='gt(empirical, test) vs standard fair [mad=%.4f]' % loss, alpha=alpha)
            filled = True

    if filled:
        if n_fair_standard_hidden is not None:
            cooc_plot_skeleton(', n_fair=%d' % n_fair_standard_hidden)
        else:
            cooc_plot_skeleton()
        
    # 6. synth vs standard/dense
    filled = False
    if synth_coocs is not None:
        if standard_coocs is not None:
            loss = cooc_loss(synth_coocs, standard_coocs[-1, ...])
            plt.scatter(synth_coocs, standard_coocs[-1, ...], label='gt(analytical) vs standard [mad=%.4f]' % loss, alpha=alpha)
            filled = True
        if dense_em_coocs is not None:
            loss = cooc_loss(synth_coocs, dense_em_coocs[-1, ...])
            plt.scatter(synth_coocs, dense_em_coocs[-1, ...], label='gt(analytical) vs dense(opt=EM) [mad=%.4f]' % loss, alpha=alpha)
            filled = True
        if standard_fair_coocs is not None:
            loss = cooc_loss(synth_coocs, standard_fair_coocs[-1, ...])
            plt.scatter(synth_coocs, standard_fair_coocs[-1, ...], label='gt(analytical) vs standard fair [mad=%.4f]' % loss, alpha=alpha)
            filled = True

    if filled:
        if n_fair_standard_hidden is not None:
            cooc_plot_skeleton(', n_fair=%d' % n_fair_standard_hidden)
        else:
            cooc_plot_skeleton()
        
    t.toc()
    
def _compute_coocs(exp_dir, n_obs, n_em, an=False, verbose=False):
    
    t = Timer()
    
    """ Get GT files """
    
    t.tic()
    
    file = '%s/data/train_X.npy' % exp_dir
    message = 'Train sequences do not exist' if verbose else None
    train_X = load_if_exists(file, message=message, retval=None)
    train_X_n_seqs = len(train_X) if train_X is not None else None
    train_X_seqlen = np.max([len(seq) for seq in train_X]) if train_X is not None else None
    
    file = '%s/data/test_X.npy' % exp_dir
    message = 'Test sequences do not exist' if verbose else None
    test_X = load_if_exists(file, message=message, retval=None)
    test_X_n_seqs = len(test_X) if test_X is not None else None
    test_X_seqlen = np.max([len(seq) for seq in test_X]) if test_X is not None else None
    
    gt_startprob, gt_transmat, gt_emissionprob = None, None, None
    if an:
        file = '%s/gt_logs/startprob.npy' % exp_dir
        message = 'GT startprob does not exist' if verbose else None
        gt_startprob = load_if_exists(file, message=message, retval=None)
        
        file = '%s/gt_logs/transmat.npy' % exp_dir
        message = 'GT transmat does not exist' if verbose else None
        gt_transmat = load_if_exists(file, message=message, retval=None)
        
        file = '%s/gt_logs/emissionprob.npy' % exp_dir
        message = 'GT emissionprob does not exist' if verbose else None
        gt_emissionprob = load_if_exists(file, message=message, retval=None)
    
    t.toc("Loaded GT files")
    
    """ Compute GT coocs """
    
    t.tic()
    
    train_coocs = empirical_coocs(train_X, n_obs)[1] if train_X is not None else None
    test_coocs = empirical_coocs(test_X, n_obs)[1] if test_X is not None else None
    
    synth_coocs = None
    if gt_startprob is not None and gt_transmat is not None and gt_emissionprob is not None:
        synth_coocs = analytical_coocs(gt_transmat, gt_emissionprob, gt_startprob).flatten()
           
    t.toc("Computed GT coocs")
    
    
    """ Load Standard & Dense EM files """
    
    t.tic()
    
    data_standard_em = {'samples': []}
    data_dense_em = {'samples': []}
    for em in range(n_em - 1, n_em): # TODO
        load_if_exists_and_append(data_standard_em, em_file(exp_dir, 'standard', em, 'aE'), verbose=verbose)
        load_if_exists_and_append(data_dense_em, em_file(exp_dir, 'dense', em, 'aE'), verbose=verbose)
        if em < n_em - 1:
            load_if_exists_and_append(data_standard_em, em_file(exp_dir, 'standard', em, 'aM'), verbose=verbose)
            load_if_exists_and_append(data_dense_em, em_file(exp_dir, 'dense', em, 'aM'), verbose=verbose)
        elif em == n_em - 1:
            load_if_exists_and_append(data_standard_em, em_file(exp_dir, 'standard', em, 'f'), verbose=verbose)
            load_if_exists_and_append(data_dense_em, em_file(exp_dir, 'dense', em, 'f'), verbose=verbose)
  
    n_em_standard_samples = len(data_standard_em['samples'])
    n_em_dense_samples = len(data_dense_em['samples'])
    
    if n_em_standard_samples < 1 or n_em_dense_samples < 1:
        raise Exception("Not enough samples")
    
    t.toc("Loaded Standard & Dense EM files")
    
    """ Compute Standard & Dense EM coocs """
    
    t.tic()
    
    standard_coocs = np.empty((n_em_standard_samples, n_obs * n_obs))
    standard_coocs.fill(np.nan)
    dense_em_coocs = np.empty((n_em_dense_samples, n_obs * n_obs))
    dense_em_coocs.fill(np.nan)
    
    for i in range(n_em_standard_samples-1, n_em_standard_samples):  # Start this loop from 0 if want to see other em steps
        sample = data_standard_em['samples'][i]
        if sample is not None:
            standard_coocs[i, ...] = empirical_coocs(sample, n_obs)[1]
        
    for i in range(n_em_dense_samples-1, n_em_dense_samples):
        sample = data_dense_em['samples'][i] # Start this loop from 0 if want to see other em steps
        if sample is not None:
            dense_em_coocs[i, ...] = empirical_coocs(sample, n_obs)[1]

    standard_coocs_valid = np.all(standard_coocs[-1, ...] != np.nan)
    dense_em_coocs_valid = np.all(dense_em_coocs[-1, ...] != np.nan)
    standard_coocs = standard_coocs if standard_coocs_valid else None
    dense_em_coocs = dense_em_coocs if dense_em_coocs_valid else None

    t.toc("Comptued Standard & Dense EM coocs")
    
    """ Load dense cooc files """
    
    t.tic()
    data_cooc = {'cooc_startprob': None, 'cooc_transmat': None, 'cooc_emissionprob': None, 'cooc_omega': None, 'cooc_sample': None}
    load_if_exists_and_append(data_cooc, cooc_file(exp_dir, 'dense'))
    t.toc("Loaded Dense cooc files")
    
    """ Compute dense cooc coocs """
    
    t.tic()
    
    dense_cooc_coocs = None
    if data_cooc['cooc_startprob'] is not None and data_cooc['cooc_transmat'] is not None and data_cooc['cooc_emissionprob'] is not None:
        dense_cooc_coocs = analytical_coocs(data_cooc['cooc_transmat'], data_cooc['cooc_emissionprob'], data_cooc['cooc_startprob']).flatten()
        
    t.toc("Computed Dense cooc coocs")
    
    """ Load fair standard if avaiable """
    #data_fair_standard = None
    #if 'compare_to_fair_standard' in exp_params and exp_params['compare_to_fair_standard']:
    data_fair_standard = {'samples': []}
    for em in range(n_em-1, n_em):  # TODO
        load_if_exists_and_append(data_fair_standard, em_file(exp_dir, 'fair_standard', em, 'aE'), verbose=verbose)
        if em < n_em - 1:
            load_if_exists_and_append(data_fair_standard, em_file(exp_dir, 'fair_standard', em, 'aM'), verbose=verbose)
        elif em == n_em - 1:
            load_if_exists_and_append(data_fair_standard, em_file(exp_dir, 'fair_standard', em, 'f'), verbose=verbose)

    n_em_fair_standard_samples = len(data_fair_standard['samples'])  # TODO what if only some files are skipped?
    
    
    """ Compute Standard fair EM coocs """
    
    t.tic()
    standard_fair_coocs, standard_fair_coocs_valid = None, None
    if n_em_fair_standard_samples > 0:
        standard_fair_coocs = np.empty((n_em_fair_standard_samples, n_obs * n_obs))
        standard_fair_coocs.fill(np.nan)
        for i in range(n_em_fair_standard_samples-1, n_em_fair_standard_samples): # Start this loop from 0 if want to see other em steps
            sample = data_fair_standard['samples'][i]
            if sample is not None:
                standard_fair_coocs[i, ...] = empirical_coocs(sample, n_obs)[1]
        
        standard_fair_coocs_valid = np.all(standard_fair_coocs[-1, ...] != np.nan)
        standard_fair_coocs = standard_fair_coocs if standard_fair_coocs_valid else None
    t.toc("Computed standard fair EM coocs")
    
    return train_coocs, test_coocs, synth_coocs, standard_coocs, dense_em_coocs, dense_cooc_coocs, standard_fair_coocs, (train_X_n_seqs, train_X_seqlen, test_X_n_seqs, test_X_seqlen)
            
def evaluate_exp_series(exp_list, use_file=False, verbose=False, ident='nl'):
    
    exp_data = dict()
    for exp_dir in exp_list:

        """ Loading parameters """

        exp_dir = str(exp_dir)

        try:
            exp_params = load_exp_params(exp_dir)
        except Exception:
            if verbose:
                print("Warning: Couldn't load %s (skipping)" % (exp_dir))
            continue

        n_em, n_em_standard, n_em_dense = _n_em(exp_params)
        n_hidden, n_gt, n_standard, n_dense = _n_hidden(exp_params)
        l, l_uz, l_vw = _l(exp_params)
        n_obs = dict_get(exp_params, 'n_emissions', default=None, cast=int)
        if n_obs is None:
            if verbose:
                print("'n_emissions' missing in exp_params dict")
            continue

        if n_dense is None or l is None:
            if verbose:
                print("Warning: Skipping %s. (No n_dense or l parameter)" % exp_dir)
            continue

        n_em_ = n_em
        if hasattr(n_em, '__len__') and len(e_em) > 1:
            n_em_ = min(n_em)

        """ Computing coocs """   

        try:

            train_coocs, test_coocs, synth_coocs, standard_coocs, dense_em_coocs, dense_cooc_coocs, standard_fair_coocs, (train_X_n_seqs, train_X_seqlen, test_X_n_seqs, test_X_seqlen) = _compute_coocs(exp_dir, n_obs, n_em_, an=True, verbose=verbose)
        except Exception:
            print("skipping %s" % exp_dir)
            continue

        n_seqs, seqlen, n_seqs_train, seqlen_train, n_seqs_test, seqlen_test = _n_seqs_seqlen(exp_params, train_X_n_seqs, train_X_seqlen, test_X_n_seqs, test_X_seqlen)

        ident_key = []
        for c in ident:
            if c == 'n':
                ident_key.append(n_hidden)
            elif c == 'l':
                ident_key.append(l)
            elif c == 'L':
                ident_key.append(seqlen)
            elif c == 'N':
                ident_key.append(n_seqs)

        ident_key = tuple(ident_key)
        if ident_key not in exp_data:
            exp_data[ident_key] = dict()

        params = dict(n_em=n_em, n_em_standard=n_em_standard, 
                      n_em_dense=n_em_dense, n_hidden=n_hidden, 
                      n_gt=n_gt, n_standard=n_standard, n_dense=n_dense, 
                      l=l, l_uz=l_uz, l_vw=l_vw, n_obs=n_obs,
                     n_seqs=n_seqs, seqlen=seqlen, n_seqs_train=n_seqs_train, n_seqs_test=n_seqs_test,
                     seqlen_train=seqlen_train, seqlen_test=seqlen_test)
        exp_data[ident_key][exp_dir] = {'params': params}
        exp_data[ident_key][exp_dir]['full_coocs'] = train_coocs, test_coocs, synth_coocs, standard_coocs, dense_em_coocs, dense_cooc_coocs, standard_fair_coocs

        cooc_mads = {}
        if synth_coocs is not None:
            if standard_coocs is not None:
                cooc_mads['gt(analytical) vs standard'] = cooc_loss(synth_coocs, standard_coocs[-1, ...])
            if dense_em_coocs is not None:
                cooc_mads['gt(analytical) vs dense(opt=EM)'] = cooc_loss(synth_coocs, dense_em_coocs[-1, ...])
            if dense_cooc_coocs is not None:
                cooc_mads['gt(analytical) vs dense(opt=cooc)'] = cooc_loss(synth_coocs, dense_cooc_coocs)
            if standard_fair_coocs is not None:
                cooc_mads['gt(analytical) vs standard fair'] = cooc_loss(synth_coocs, standard_fair_coocs[-1, ...])

        if train_coocs is not None:
            if standard_coocs is not None:
                cooc_mads['gt(train) vs standard'] = cooc_loss(train_coocs, standard_coocs[-1, ...])
            if dense_em_coocs is not None:
                cooc_mads['gt(train) vs dense(opt=EM)'] = cooc_loss(train_coocs, dense_em_coocs[-1, ...])
            if dense_cooc_coocs is not None:
                cooc_mads['gt(train) vs dense(opt=cooc)'] = cooc_loss(train_coocs, dense_cooc_coocs)
            if standard_fair_coocs is not None:
                cooc_mads['gt(train) vs standard fair'] = cooc_loss(train_coocs, standard_fair_coocs[-1, ...])

        if test_coocs is not None:
            if standard_coocs is not None:
                cooc_mads['gt(test) vs standard'] = cooc_loss(test_coocs, standard_coocs[-1, ...])
            if dense_em_coocs is not None:
                cooc_mads['gt(test) vs dense(opt=EM)'] = cooc_loss(test_coocs, dense_em_coocs[-1, ...])
            if dense_cooc_coocs is not None:
                cooc_mads['gt(test) vs dense(opt=cooc)'] = cooc_loss(test_coocs, dense_cooc_coocs)
            if standard_fair_coocs is not None:
                cooc_mads['gt(test) vs standard fair'] = cooc_loss(test_coocs, standard_fair_coocs[-1, ...])

        exp_data[ident_key][exp_dir]['cooc_MAD'] = cooc_mads

        """ Reading losses/loglikes """

        for model in ('standard', 'dense', 'fair_standard'):

            if model not in exp_data[ident_key][exp_dir]:
                exp_data[ident_key][exp_dir][model] = dict()

            params_key = '%s_params' % model

            data_em = {'loglike': None, 'val_loglike': None, 'train_losses': None, 'test_losses': None, 'test_gamma_losses': None}
            params = dict_get(exp_params, params_key, default=None, cast=dict)
            n_em = dict_get(params, 'em_iter', default=None, cast=int)

            if n_em is None:
                if verbose:
                    print("Warning: Skipping model %s for experiment %s. (No n_em parameter)" % (model, exp_dir))
                continue

            load_if_exists_and_append(data_em, em_file(exp_dir, model, n_em - 1, 'f'), verbose=verbose)
            exp_data[ident_key][exp_dir][model].update(data_em)

            if model == 'dense':
                data_coocs = {'cooc_logprobs': None, 'cooc_val_logprobs': None}
                load_if_exists_and_append(data_coocs, cooc_file(exp_dir, 'dense'))
                exp_data[ident_key][exp_dir][model].update(data_coocs)

    if use_file and type(use_file) == str:
        np.savez_compressed(use_file, exp_data=exp_data)
        
    return exp_data
    
def plot_exp_data(exp_data, ident_str='nl', ident_val=None, plot_path=None, gt_ident='test'):
    
    if exp_data is None:
          raise Exception("Given exp_data cannot be None")
    elif type(exp_data) == str:
        exp_data = np.load(exp_data, allow_pickle=True)['exp_data'].item()
    elif type(exp_data) != dict:
        raise Exception("Given exp_data has to be either str or dict")
    
    gt_ident = str(gt_ident)
    if gt_ident not in ('test', 'analytical'):
        raise Exception("Given gt ident has to be either 'test' or 'analytical'")
    
    """ Plot l/n MADs """    
    
    mads_median = ([], [], [], [])
    mads_25_quant = ([], [], [], [])
    mads_75_quant = ([], [], [], [])

    ident_str_ = ident_str
    ident_candidates = np.array(exp_data.keys())
    ident_idxs_ = list(range(ident_candidates.shape[1]))
    if ident_val is not None:
        if type(ident_val) == list:
            ident_idxs = list(range(len(ident_val)))
            ident_idxs_ = list(range(len(ident_val), ident_candidates.shape[1]))
            ident_list = ident_val
            ident_str_ = ident_str[:len(ident_val)]
        elif type(ident_val) == dict:
            if ident_candidates.shape[1] != len(ident_str):
                raise Exception("Can only use ident_val as dict if ident_str is as long as the keys")
            
            ident_idxs = []
            ident_idxs_ = []
            ident_list = []
            ident_str_ = ''
            for i, c in enumerate(ident_str):
                if c in ident_val.keys():
                    ident_idxs.append(i)
                    ident_list.append(ident_val[c])
                else:
                    ident_idxs_.append(i)
                    ident_str_ += c
        
        ident_candidates = ident_candidates[np.all(ident_candidates[:, ident_idxs] == ident_list, axis=1)]
    
    sorted_idents = ident_candidates[np.lexsort(ident_candidates[:,::-1].T)]
    for ident in sorted_idents:
        ident = tuple(ident)
        mads = ([], [], [], [])
        for key in exp_data[ident].keys():
            
            mads[0].append(exp_data[ident][key]['cooc_MAD']['gt(%s) vs standard' % gt_ident] * 1000)
            mads[1].append(exp_data[ident][key]['cooc_MAD']['gt(%s) vs dense(opt=EM)' % gt_ident] * 1000)
            mads[2].append(exp_data[ident][key]['cooc_MAD']['gt(%s) vs dense(opt=cooc)' % gt_ident] * 1000)
            mads[3].append(exp_data[ident][key]['cooc_MAD']['gt(%s) vs standard fair' % gt_ident] * 1000)

        medians = np.median(mads, axis=1)
        mads_median[0].append(medians[0])
        mads_median[1].append(medians[1])
        mads_median[2].append(medians[2])
        mads_median[3].append(medians[3])

        quant_25 = np.quantile(mads, 0.25, axis=1)
        mads_25_quant[0].append(quant_25[0])
        mads_25_quant[1].append(quant_25[1])
        mads_25_quant[2].append(quant_25[2])
        mads_25_quant[3].append(quant_25[3])

        quant_75 = np.quantile(mads, 0.75, axis=1)
        mads_75_quant[0].append(quant_75[0])
        mads_75_quant[1].append(quant_75[1])
        mads_75_quant[2].append(quant_75[2])
        mads_75_quant[3].append(quant_75[3])

    colors = ['orange', 'red', 'green', 'blue']

    fig, ax = plt.subplots()
    x = range(len(sorted_idents))
    mads_median = np.array(mads_median)
    
    start = 0
    cur_n = 0
    n_idx_in_ident = ident_str.find('n')
    for stop, _ident in enumerate(sorted_idents):
        n = _ident[n_idx_in_ident]
        if cur_n == 0:
            cur_n  = n
        elif stop == len(sorted_idents) - 1:
            stop += 1
            ax.plot(x[start:stop], mads_median[1, start:stop], marker='x', label='Dense EM', color=colors[1], alpha=0.8)
            ax.plot(x[start:stop], mads_median[2, start:stop], marker='x', label='Dense direct', color=colors[2], alpha=0.8)
            ax.plot(x[start:stop], mads_median[3, start:stop], marker='x', label='Standard fair', color=colors[3], alpha=0.8)
            
            ax.plot(x[start:stop], mads_median[0, start:stop], marker='x', label='Standard', color=colors[0], alpha=0.8)
            
        elif n > cur_n or stop == len(sorted_idents):
            ax.plot(x[start:stop], mads_median[0, start:stop], marker='x', color=colors[0], alpha=0.8)
            ax.plot(x[start:stop], mads_median[1, start:stop], marker='x', color=colors[1], alpha=0.8)
            ax.plot(x[start:stop], mads_median[2, start:stop], marker='x', color=colors[2], alpha=0.8)
            ax.plot(x[start:stop], mads_median[3, start:stop], marker='x', color=colors[3], alpha=0.8)
            start = stop
            cur_n = n

    for i in x:
        ax.plot([i, i], [mads_25_quant[0][i], mads_75_quant[0][i]], marker='_', color=colors[0],linewidth=4, markersize=18, markeredgewidth=3, alpha=0.5)
        ax.plot([i, i], [mads_25_quant[1][i], mads_75_quant[1][i]], marker='_', color=colors[1],linewidth=4, markersize=18, markeredgewidth=3, alpha=0.5)
        ax.plot([i, i], [mads_25_quant[2][i], mads_75_quant[2][i]], marker='_', color=colors[2],linewidth=4, markersize=18, markeredgewidth=3, alpha=0.5)
        ax.plot([i, i], [mads_25_quant[3][i], mads_75_quant[3][i]], marker='_', color=colors[3],linewidth=4, markersize=18, markeredgewidth=3, alpha=0.5)

    ax.legend()
    ax.set_ylim([0, 1.1*np.max(mads_median)])
    ax.set_ylabel(r'Cooc MAD ($10^{-3}$)')
    ax.set_xlabel('[' + ','.join(str(c) for c in ident_str_) + ']')
    ax.set_xticks(x)
    ax.set_xticklabels(['[' + ','.join(str(c) for c in ident) + ']' for ident in sorted_idents[:, ident_idxs_]])
    fig.set_size_inches(19, 10)
    plt.xticks(rotation=45)
    if plot_path is not None:
        plt.savefig(plot_path + '/coocmad.png', bbox_inches='tight')
    fig.show()
    
    """ Plot l/n loglikes """
    
    loglikes_median = ([], [], [], [])
    loglikes_25_quant = ([], [], [], [])
    loglikes_75_quant = ([], [], [], [])

    for ident in sorted_idents:
        ident = tuple(ident)

        loglikes = ([], [], [], [])
        for key in exp_data[ident].keys():
            n_seqs = exp_data[ident][key]['params']['n_seqs']
            seqlen = exp_data[ident][key]['params']['seqlen']
            loglikes[0].append(exp_data[ident][key]['standard']['val_loglike'] / (n_seqs*seqlen))
            loglikes[1].append(exp_data[ident][key]['dense']['val_loglike']/ (n_seqs*seqlen))
            loglikes[2].append(np.sum(exp_data[ident][key]['dense']['cooc_val_logprobs'])/ (n_seqs*seqlen))
            loglikes[3].append(exp_data[ident][key]['fair_standard']['val_loglike']/ (n_seqs*seqlen))
            
        loglikes = -np.array(loglikes)
        medians = np.median(loglikes, axis=1)
        loglikes_median[0].append(medians[0])
        loglikes_median[1].append(medians[1])
        loglikes_median[2].append(medians[2])
        loglikes_median[3].append(medians[3])

        quant_25 = np.quantile(loglikes, 0.25, axis=1)
        loglikes_25_quant[0].append(quant_25[0])
        loglikes_25_quant[1].append(quant_25[1])
        loglikes_25_quant[2].append(quant_25[2])
        loglikes_25_quant[3].append(quant_25[3])

        quant_75 = np.quantile(loglikes, 0.75, axis=1)
        loglikes_75_quant[0].append(quant_75[0])
        loglikes_75_quant[1].append(quant_75[1])
        loglikes_75_quant[2].append(quant_75[2])
        loglikes_75_quant[3].append(quant_75[3])
    
    loglikes_median = np.array(loglikes_median)

    fig, ax = plt.subplots()
    x = range(len(sorted_idents))
    
    start = 0
    cur_n = 0
    for stop, _ident in enumerate(sorted_idents):
        n = _ident[n_idx_in_ident]
        if cur_n == 0:
            cur_n  = n
        elif stop == len(sorted_idents) - 1:
            stop += 1
            ax.plot(x[start:stop], loglikes_median[1, start:stop], marker='x', label='Dense EM', color=colors[1], alpha=0.8)
            ax.plot(x[start:stop], loglikes_median[2, start:stop], marker='x', label='Dense direct', color=colors[2], alpha=0.8)
            ax.plot(x[start:stop], loglikes_median[3, start:stop], marker='x', label='Standard fair', color=colors[3], alpha=0.8)
            
            ax.plot(x[start:stop], loglikes_median[0, start:stop], marker='x', label='Standard', color=colors[0], alpha=0.8)
        
        if n > cur_n or stop == len(sorted_idents):
            ax.plot(x[start:stop], loglikes_median[0, start:stop], marker='x', color=colors[0], alpha=0.8)
            ax.plot(x[start:stop], loglikes_median[1, start:stop], marker='x', color=colors[1], alpha=0.8)
            ax.plot(x[start:stop], loglikes_median[2, start:stop], marker='x', color=colors[2], alpha=0.8)
            ax.plot(x[start:stop], loglikes_median[3, start:stop], marker='x', color=colors[3], alpha=0.8)
            start = stop
            cur_n = n
            
    for i in x:
        ax.plot([i, i], [loglikes_25_quant[0][i], loglikes_75_quant[0][i]], marker='_', color=colors[0], linewidth=4, markersize=18, markeredgewidth=3,  alpha=0.5)
        ax.plot([i, i], [loglikes_25_quant[1][i], loglikes_75_quant[1][i]], marker='_', color=colors[1], linewidth=4, markersize=18,markeredgewidth=3, alpha=0.5)
        ax.plot([i, i], [loglikes_25_quant[2][i], loglikes_75_quant[2][i]], marker='_', color=colors[2], linewidth=4, markersize=18, markeredgewidth=3, alpha=0.5)
        ax.plot([i, i], [loglikes_25_quant[3][i], loglikes_75_quant[3][i]], marker='_', color=colors[3], linewidth=4, markersize=18, markeredgewidth=3,alpha=0.5)

    ax.legend()
    ax.set_ylim([0, 1.1*np.max(loglikes_median)])
    ax.set_ylabel('normalized NLL')
    ax.set_xlabel('[' + ','.join(str(c) for c in ident_str_) + ']')
    ax.set_xticks(x)
    ax.set_xticklabels(['[' + ','.join(str(c) for c in ident) + ']' for ident in sorted_idents[:, ident_idxs_]])
    fig.set_size_inches(19, 10)
    plt.xticks(rotation=45)
    if plot_path is not None:
        plt.savefig(plot_path + '/nll.png', bbox_inches='tight')
    fig.show()
        
    return exp_data
       
def plot(exp_dir, verbose=False):
    exp_params = load_exp_params(exp_dir)
    plot_loss_and_loglike(exp_params, 'standard', verbose)
    plot_loss_and_loglike(exp_params, 'dense', verbose)
    plot_coocurrences(exp_params, verbose, an=True)

    
