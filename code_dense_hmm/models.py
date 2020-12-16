from hmmlearn.hmm import MultinomialHMM
from hmmlearn.base import ConvergenceMonitor, iter_from_X_lengths, log_mask_zero, check_is_fitted, check_array#_utils
import tensorflow as tf
import numpy as np

from hmmc import _hmmc as _hmmcmod #mod
from utils import check_arr, pad_to_seqlen, check_random_state, dict_get, check_dir, compute_stationary, empirical_coocs

import time


class HMMLoggingMonitor:
    
    def __init__(self, log_config=None):
        
        # Default log_config
        self.log_config = {
                'exp_folder': None,  # root experiment folder
                #'plot_folder': None, # folder to store plots in
                'log_folder': None, # folder to store array-data in
                'metrics_initial' : True,  # whether to compute metrics before first estep
                'metrics_after_mstep_every_n_iter': 1, # frequency of computing metrics after mstep or none
                'metrics_after_estep_every_n_iter': 1, # frequency of computing metrics after estep or none
                'metrics_after_convergence': True, # whether to compute metrics after the training
                'gamma_after_estep': False, # whether to compute gammas (they are very large!)
                'gamma_after_mstep': False,
                'test_gamma_after_estep': False,  # whether to compute test gammas ..
                'test_gamma_after_mstep': False,
                'samples_after_estep': None,  # (n_seqs, seqlen) sample to draw after estep or None
                'samples_after_mstep': None,   # (n_seqs, seqlen) sample to draw after mstep or None
                'samples_after_cooc_opt': None # (n_seqs, seqlen) sample to draw after fitting model's coocs
                }
        
        if log_config is not None:
            self.log_config.update(dict(log_config))
    
    def _check_log_path(self):
        log_conf = self.log_config
        exp_folder = '.' if log_conf['exp_folder'] is None else log_conf['exp_folder']
        log_folder = '/data' if log_conf['log_folder'] is None else log_conf['log_folder']
        log_path = check_dir(exp_folder + log_folder)
        return log_path
        
    def log(self, file_name, log_dict, key_func=None):#stats, em_iter, ident):
        log_path = self._check_log_path()
        self._log(log_path, file_name, log_dict, key_func)
        
    def _log(self, log_path, file_name, log_dict, key_func=None):
        if key_func is not None:
            np.savez_compressed(log_path + '/' + file_name, **{key_func(key): log_dict[key] for key in log_dict.keys()})
        else:
            np.savez_compressed(log_path + '/' + file_name, **log_dict)

    def emname(self, em_iter, ident):
        return "logs_em=%d_ident=%s" % (int(em_iter), ident)
    
class GammaMultinomialHMM(MultinomialHMM):
            
        
    """ Base class for Hidden Markov Model of hmmlearn extended for computing all gamma values and logging """
    def __init__(self, n_hidden_states=1, n_observables=None,
                 startprob_prior=1.0, transmat_prior=1.0, 
                 random_state=None, em_iter=10, convergence_tol=1e-2, verbose=False,
                params="ste", init_params="ste", logging_monitor=None):
    
        self.matrix_initializer = None
        if init_params is None:
            init_params = "ste"
        elif callable(init_params):
            self.matrix_initializer = init_params
            init_params = ''
        
        super(GammaMultinomialHMM, self).__init__(n_components=n_hidden_states,
                                startprob_prior=startprob_prior,
                                transmat_prior=transmat_prior,
                                algorithm="viterbi",
                                random_state=random_state,
                                n_iter=em_iter,
                                tol=convergence_tol,
                                verbose=verbose,
                                params=params,
                                init_params=init_params)
        
        self.n_observables = n_observables
        self.logging_monitor = logging_monitor if logging_monitor is not None else HMMLoggingMonitor()
        
        if self.matrix_initializer is not None and self.n_observables is not None:
            self._init_matrices_using_initializer(self.matrix_initializer)
        
    def _init_gammas(self, n_seqs, max_seqlen):
        gamma = np.zeros((n_seqs, max_seqlen, self.n_components))
        bar_gamma = np.zeros((max_seqlen, self.n_components))
        gamma_pairwise = np.zeros((n_seqs, max_seqlen-1, self.n_components, self.n_components))
        bar_gamma_pairwise = np.zeros((max_seqlen-1, self.n_components, self.n_components))
        return gamma, bar_gamma, gamma_pairwise, bar_gamma_pairwise
        
    def _initialize_sufficient_statistics(self, n_seqs, max_seqlen):
        """ Initialize a dictionary holding:
            
            - nobs: int; Number of samples in the data processed so far
            - start: array, shape (n_hidden_states,); 
                     An array where the i-th element corresponds to the posterior
                     probability of the first sample being generated by the i-th 
                     state.
            - trans: array, shape (n_hidden_states, n_hidden_states);
            An array where the (i, j)-th element corresponds to the
            posterior probability of transitioning between the i-th to j-th
            states.
            - obs: array, shape (n_hidden_states, n_observables);
            An array where the (i, j)-th element corresponds to the ...
            
            - max_seqlen: int; Maximum sequence length in given data
            - n_seqs: int; Number of sequences in given data
            - gamma: array, shape (n_seqs, max_seqlen, n_hidden_states);
            The posterior probabilities w.r.t. every observation sequence
            - bar_gamma: array, shape (n_seqs, max_seqlen, n_hidden_states);
            Cumulative posterior probabilities (Sum of gamma over first dimension)
            - gamma_pairwise: array, shape (n_seqs, max_seqlen-1, n_hidden_states, n_hidden_states);
            Pairwise gamma terms over all observation sequences and times
            - bar_gamma_pairwise: array, shape (max_seqlen-1, n_hidden_states, n_hidden_states);
            Cumulative pairwise gamma terms (Sum of gamma_pairwise over first dim.)
            
            This function is called before every em-iteration.
        """
        
        stats = super(GammaMultinomialHMM, self)._initialize_sufficient_statistics()
        
        stats['max_seqlen'] = max_seqlen
        stats['n_seqs'] = n_seqs
        
        stats['gamma'], stats['bar_gamma'], \
        stats['gamma_pairwise'], stats['bar_gamma_pairwise'] = self._init_gammas(n_seqs, max_seqlen)
        
        stats['all_logprobs'] = np.zeros((n_seqs,))
        
        return stats
    
    """ Initializes the n_features (number of observables),
        checks input data for Multinomial distribution and
        initializes Matrices
    """ 
    def _init(self, X, lengths=None):
        
        X, n_seqs, max_seqlen = check_arr(X, lengths)
        
        # This initializes the transition matrices: 
        # startprob_ and transmat_ to 1/n_hidden_states
        # emissionprob_ to randomly chosen distributions
        # and sets self.n_features to the number of unique symbols in X
        # and checks that X are samples of a multinomial distribution
        super(GammaMultinomialHMM, self)._init(X, lengths=lengths)
        
        if self.n_observables is None:
            self.n_observables = self.n_features
        elif self.n_features != self.n_observables:
            raise Exception("n_observables was given %d, but given data has only"
                            "%d unique symbols" % (self.n_observables, self.n_features))
        
        if self.matrix_initializer is not None:
            self._init_matrices_using_initializer(self.matrix_initializer)
            
        return X, n_seqs, max_seqlen
            
    def _init_matrices_using_initializer(self, matrix_initializer):
        # If random_state is None, this returns a new np.RandomState instance
        # If random_state is int, a new np.RandomState instance seeded with random_state 
        # is returned. If random_state is already an instance of np.RandomState, 
        # it is returned
        self.random_state = check_random_state(self.random_state)

        pi, A, B = matrix_initializer(self.n_components, self.n_observables, self.random_state)
        self.startprob_, self.transmat_ = pi, A
        self.emissionprob_ = B
        self._check()
                
    def fit(self, X, lengths=None, val=None, val_lengths=None):
        """Estimate model parameters.
        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        Returns
        -------
        self : object
            Returns self.
        """
        
        # Initializes the n_features (number of observables),
        # checks input data for Multinomial distribution and
        # initializes matrices
        X, n_seqs, max_seqlen = self._init(X, lengths=lengths)
        
        # This makes sure that transition matrices have the correct shape
        # and represent distributions
        self._check()
        
        log_config = self.logging_monitor.log_config
        emname = self.logging_monitor.emname
        self.monitor_._reset()
        for iter in range(self.n_iter):
            
            stats = self._initialize_sufficient_statistics(n_seqs, max_seqlen)
            
            # Compute metrics before first E-step / after M-step
            if iter == 0 and log_config['metrics_initial']:
                log_dict = self._compute_metrics(X, lengths, stats, iter, 'i', val, val_lengths)
                self.logging_monitor.log(emname(iter, 'i'), log_dict)
            
            # Do E-step
            stats, total_logprob = self._forward_backward_gamma_pass(X, lengths, stats)

            # Compute metrics after E-step
            if log_config['metrics_after_estep_every_n_iter'] is not None:
                if iter % log_config['metrics_after_estep_every_n_iter'] == 0:
                    log_dict = self._compute_metrics(X, lengths, stats, iter, 'aE', val, val_lengths)
                    self.logging_monitor.log(emname(iter, 'aE'), log_dict)

            # XXX must be before convergence check, because otherwise
            #     there won't be any updates for the case ``n_iter=1``.
            self._do_mstep(stats)

            if log_config['metrics_after_mstep_every_n_iter'] is not None:
                if iter % log_config['metrics_after_mstep_every_n_iter'] == 0:
                    log_dict = self._compute_metrics(X, lengths, stats, iter, 'aM', val, val_lengths)
                    self.logging_monitor.log(emname(iter, 'aM'), log_dict)

            self.monitor_.report(total_logprob)
            #if self.monitor_.converged:
            #    print("Exiting EM early ... (convergence tol)")
            #    break
            
        # Final metrics
        if log_config['metrics_after_convergence']:
            log_dict = self._compute_metrics(X, lengths, stats, iter, 'f', val, val_lengths)
            self.logging_monitor.log(emname(iter, 'f'), log_dict)

        return self
    
    def _forward_backward_gamma_pass(self, X, lengths=None, stats=None, params=None):
        
        params = self.params if params is None else params
        
        if stats is None:
            X, n_seqs, max_seqlen = check_arr(X)
            stats = self._initialize_sufficient_statistics(n_seqs, max_seqlen)
        
        total_logprob = 0
        
        # Iterate over all sequences
        for seq_idx, (i, j) in enumerate(iter_from_X_lengths(X, lengths)):
            
            stats['nobs'] = seq_idx

            # Compute posteriors by forward-backward algorithm
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprob, fwdlattice = self._do_forward_pass(framelogprob)
            
            stats['all_logprobs'][seq_idx] = logprob # logprob = probability of X[i:j]
            
            total_logprob += logprob
            bwdlattice = self._do_backward_pass(framelogprob)
            posteriors = self._compute_posteriors(fwdlattice, bwdlattice)
            
            # Pad posteriors with zeros such that its length equals max_seqlen
            posteriors = pad_to_seqlen(posteriors, stats['max_seqlen'])

            n_samples, n_components = framelogprob.shape
            # when the sample is of length 1, it contains no transitions
            # so there is no reason to update our trans. matrix estimate
            if n_samples <= 1:
                continue

            # Compute pairwise gammas and log_xi_sum
            cur_gamma_pairwise = np.zeros_like(stats['bar_gamma_pairwise'])
            log_xi_sum = np.full((n_components, n_components), -np.inf)
            _hmmcmod._compute_log_xi_sum(n_samples, n_components, fwdlattice,
                                      log_mask_zero(self.transmat_),
                                      bwdlattice, framelogprob,
                                      log_xi_sum, cur_gamma_pairwise)
            
            
            # Compute gammas
            stats['gamma'][seq_idx, :, :] = posteriors
            stats['bar_gamma'] += posteriors
            stats['bar_gamma_pairwise'] += cur_gamma_pairwise
            stats['gamma_pairwise'][seq_idx, :, :, :] = cur_gamma_pairwise

            if 's' in self.params:
                stats['start'] += posteriors[0]
            if 't' in self.params:
                with np.errstate(under="ignore"):
                    stats['trans'] += np.exp(log_xi_sum)
            if 'e' in self.params:
                for t, symbol in enumerate(np.concatenate(X[i:j])):
                    stats['obs'][:, symbol] += posteriors[t]
        
        return stats, total_logprob

    
    # Currently supported: Total log-likelihood on given data, 
    # individual log-likelihoods 
    def _compute_metrics(self, X, lengths, stats, em_iter, ident, 
                         val=None, val_lengths=None):
        
        
        log_config = self.logging_monitor.log_config
        log_dict = {}
        stats = stats.copy()
        
        # log_dict shall contain:
        # After E and M:
        # loglike, all_loglikes
        # val_loglike, val_all_loglikes
        # gamma, bar_gamma, gamma_pairwise, bar_gamma_pairwise
        # startprob_, transmat_, emissionprob_
        # If parameter is set, after E and M: samples
        # If parameter is set, after E:
        # val_gamma, val_bar_gamma, val_gamma_pairwise, val_bar_gamma_pairwise
        # If parameter is set, after M:
        # val_gamma, val_bar_gamma, val_gamma_pairwise, val_bar_gamma_pairwise
        
        # Gamma and transition matrices
        log_dict['startprob'], log_dict['transmat'], log_dict['emissionprob'] = self.startprob_, self.transmat_, self.emissionprob_
        
        # Get log-likelihoods on training set
        if ident == 'aE':
            
            if log_config['gamma_after_estep']:
                log_dict['gamma'], log_dict['bar_gamma'] = stats['gamma'], stats['bar_gamma']
                log_dict['gamma_pairwise'], log_dict['bar_gamma_pairwise'] = stats['gamma_pairwise'], stats['bar_gamma_pairwise']
            
            # After the estep, you can get the current log-likelihoods from the stats dict
            log_dict['loglike'] = np.sum(stats['all_logprobs'])
            log_dict['all_loglikes'] = stats['all_logprobs']
            
            # DEBUG
            #log_dict['all_loglikes'] = self.score_individual_sequences(X, lengths)[0]
            #log_dict['loglike'] = np.sum(log_dict['all_loglikes']) #self.score(X, lengths)
            
            #print("In _compute_metrics aE", log_dict['all_loglikes'].shape, stats['all_logprobs'].shape)
            #print('stats sum:', np.sum(stats['all_logprobs']), log_dict['loglike'])
            #_al = self.score_individual_sequences(X, lengths)[0]
            #print(np.sum(stats['all_logprobs']) == np.sum(_al))
            #print(np.all(stats['all_logprobs'] == _al))
            #print('stats all logprobs', stats['all_logprobs'])
            #print('all loglikes', log_dict['all_loglikes'])
            #_scored_loglikes = np.array(log_dict['all_loglikes'])
            #_stat_loglikes = np.array(stats['all_logprobs'])
            #print(np.all(_scored_loglikes == _stat_loglikes))
            #print(np.sum(_scored_loglikes) == np.sum(_stat_loglikes))
            
            
            if log_config['samples_after_estep'] is not None:
                sample_sizes = None
                if type(log_config['samples_after_estep']) == tuple:
                    sample_sizes = log_config['samples_after_estep']
                else:
                    if val_lengths is not None:
                        sample_sizes = (len(val_lengths), np.max(val_lengths))
                    else:
                        sample_sizes = (stats['n_seqs'], stats['max_seqlen'])
                log_dict['samples'] = self.sample_sequences(*sample_sizes)
        
        elif ident == 'aM' or ident == 'f' or ident == 'i':
            
            # After the mstep, the stats dict contains the log-likelihoods under previous transition matrices
            # Therefore, cannot use stats dict
            log_dict['all_loglikes'] = self.score_individual_sequences(X, lengths)[0]
            log_dict['loglike'] = np.sum(log_dict['all_loglikes']) #self.score(X, lengths)
            
            if log_config['gamma_after_mstep']:
                log_dict['gamma'], log_dict['bar_gamma'] = stats['gamma'], stats['bar_gamma']
                log_dict['gamma_pairwise'], log_dict['bar_gamma_pairwise'] = stats['gamma_pairwise'], stats['bar_gamma_pairwise']
            
            if log_config['samples_after_mstep'] is not None:
                sample_sizes = None
                if type(log_config['samples_after_mstep']) == tuple:
                    sample_sizes = log_config['samples_after_mstep']
                else:
                    if val_lengths is not None:
                        sample_sizes = (len(val_lengths), np.max(val_lengths))
                    else:
                        sample_sizes = (stats['n_seqs'], stats['max_seqlen'])
                log_dict['samples'] = self.sample_sequences(*sample_sizes)
        
        # Get log-likelihoods and gammas on test set
        if val is not None:
            
            if log_config['test_gamma_after_estep'] and ident == 'aE' or log_config['test_gamma_after_mstep'] and ident == 'aM':
                
                val_stats, val_loglike = self._forward_backward_gamma_pass(val, val_lengths)
                log_dict['val_all_loglikes'] = val_stats['all_logprobs']
                log_dict['val_loglike'] = val_loglike

                # Gammas
                log_dict['val_gamma'], log_dict['val_bar_gamma'] = val_stats['gamma'], val_stats['bar_gamma']
                log_dict['val_gamma_pairwise'] = val_stats['val_gamma_pairwise']
                log_dict['val_bar_gamma_pairwise'] = val_stats['val_bar_gamma_pairwise']
            
            else:
                log_dict['val_all_loglikes'], log_dict['val_loglike'] = self.score_individual_sequences(val, val_lengths)
        
        return log_dict
    
    """ Sample n_seqs (int) sequences each of length seqlen (int). Returns an array of shape (n_seqs, seqlen, n_features) """
    def sample_sequences(self, n_seqs, seqlen):
        # hmm.sample returns (sequence, state_sequence)
        return np.array([self.sample(n_samples=seqlen)[0] for _ in range(n_seqs)])
    
    def score_individual_sequences(self, X, lengths=None):
        """Compute the log probability under the model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        Returns
        -------
        logprob : float
            Log likelihood of ``X``.
        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        #_utils.check_is_fitted(self, "startprob_")
        check_is_fitted(self, "startprob_")
        self._check()
        
        logprobs = None
        if lengths is None:
            logprobs = np.zeros(1)
        else:
            logprobs = np.zeros(len(lengths))
            
        X = check_array(X)
        # XXX we can unroll forward pass for speed and memory efficiency.
        logprob = 0
        for seq_idx, (i, j) in enumerate(iter_from_X_lengths(X, lengths)):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij, _fwdlattice = self._do_forward_pass(framelogprob)
            logprobs[seq_idx] = logprobij
            logprob += logprobij
        return logprobs, logprob
    
    """ Turns the given observations X of shape (n_sequences, 1)
        into an observations matrix (n_sequences, max_seqlen) 
        by padding sequences to max_seqlen """
    def _observations_to_padded_matrix(self, X, lengths):
        
        O, n_seqs, max_seqlen = check_arr(X, lengths)
        O = O.flatten()
        
        # X has shape (seqs, 1); 
        # Turn it into (seqs, max_seqlen) by padding
        arr = np.zeros((len(lengths), max_seqlen))
        for idx, (i, j) in enumerate(iter_from_X_lengths(X, lengths)):
            sequence = O[i:j]
            arr[idx] = np.pad(sequence, 
                              (0, max_seqlen - len(sequence)), 
                              'constant', constant_values=(0))
        
        # Check if arr contains only integer 
        if not np.all(np.equal(np.mod(arr, 1), 0)):
            raise Exception("Sequence elements have to be integer! \n"
                            "arr: \n %s \n X \n %s" % (str(arr), str(O)))
        O = arr.astype(np.int)
        return O, n_seqs, max_seqlen
                                

class StandardHMM(GammaMultinomialHMM):
        
    def __init__(self, n_hidden_states=1, n_observables=None,
             startprob_prior=1.0, transmat_prior=1.0, 
             random_state=None, em_iter=10, convergence_tol=1e-2, verbose=False,
            params="ste", init_params="ste", logging_monitor=None):
    
        super(StandardHMM, self).__init__(n_hidden_states=n_hidden_states,
                                     n_observables=n_observables,
                                    startprob_prior=startprob_prior,
                                    transmat_prior=transmat_prior,
                                    random_state=random_state,
                                    em_iter=em_iter,
                                    convergence_tol=convergence_tol,
                                    verbose=verbose,
                                    params=params,
                                    init_params=init_params,
                                    logging_monitor=logging_monitor)
        
    def _compute_metrics(self, X, lengths, stats, em_iter, ident, 
                         val=None, val_lengths=None):
        
        log_config = self.logging_monitor.log_config
        log_dict = super(StandardHMM, self)._compute_metrics(X, lengths, stats, em_iter, ident, val, val_lengths)
        
        gamma, bar_gamma = stats['gamma'], stats['bar_gamma']
        bar_gamma_pairwise = stats['bar_gamma_pairwise']
        log_dict['train_losses'] = self._compute_loss(X, lengths, bar_gamma, bar_gamma_pairwise, gamma)
        
        if val is not None:
            
            log_dict['test_losses'] = self._compute_loss(val, val_lengths, bar_gamma, bar_gamma_pairwise, gamma)
            
            # val-gammas are not necessarily in log_dict
            val_bar_gamma = dict_get(log_dict, 'val_bar_gamma')
            val_bar_gamma_pairwise = dict_get(log_dict, 'val_bar_gamma_pairwise')
            val_gamma = dict_get(log_dict, 'val_gamma')
            
            if val_bar_gamma is not None and val_bar_gamma_pairwise is not None and val_gamma is not None:
                log_dict['test_gamma_losses'] = self._compute_loss(val, val_lengths, val_bar_gamma, val_bar_gamma_pairwise, val_gamma)
    
        return log_dict
    
    """ Computes loss on given sequence; Using given gamma terms and
    the current transition matrices
    """
    def _compute_loss(self, X, lengths, bar_gamma, bar_gamma_pairwise, gamma):
        
        X, n_seqs, max_seqlen = self._observations_to_padded_matrix(X, lengths)
        
        log_A = np.log(self.transmat_)
        log_B = np.log(self.emissionprob_)
        log_pi = np.log(self.startprob_)
        
        tilde_B = log_B[:, X] # Has shape (n_hidden_states, seqs, max_seqlen)
        
        loss1 = -np.einsum('s,s->', log_pi, bar_gamma[0, :])
        loss2 = -np.einsum('jl,tjl->', log_A, bar_gamma_pairwise)
        loss3 = -np.einsum('jit,itj->', tilde_B, gamma) 
        loss = loss1 + loss2 + loss3
        
        return np.array([loss, loss1, loss2, loss3])
    

class DenseHMM(GammaMultinomialHMM):
    
    SUPPORTED_REPRESENTATIONS = frozenset(('uvwzz0', 'vzz0'))
    SUPPORTED_OPT_SCHEMES = frozenset(('em', 'cooc'))
    
    def __init__(self, n_hidden_states=1, n_observables=None,
             startprob_prior=1.0, transmat_prior=1.0, 
             random_state=None, em_iter=10, convergence_tol=1e-2, verbose=False,
            params="ste", init_params="ste", logging_monitor=None, mstep_config=None,
                opt_schemes=None):
    
        super(DenseHMM, self).__init__(n_hidden_states=n_hidden_states,
                                     n_observables=n_observables,
                                    startprob_prior=startprob_prior,
                                    transmat_prior=transmat_prior,
                                    random_state=random_state,
                                    em_iter=em_iter,
                                    convergence_tol=convergence_tol,
                                    verbose=verbose,
                                    params=params,
                                    init_params=init_params,
                                    logging_monitor=logging_monitor)
        
        self.graph = None
        
        mstep_config = {} if mstep_config is None else mstep_config
        self.opt_schemes = self.SUPPORTED_OPT_SCHEMES if opt_schemes is None else set(opt_schemes)
                     
        # Used for both optimization schemes
        self.initializer = dict_get(mstep_config, 'initializer', default=tf.initializers.random_normal(0.,1.))
        self.l_uz = dict_get(mstep_config, 'l_uz', default=3)
        self.l_vw = dict_get(mstep_config, 'l_vw', default=3)
        self.trainables = dict_get(mstep_config, 'trainables', default='uvwzz0')
        self.representations = dict_get(mstep_config, 'representations', default='uvwzz0')
        self.kernel = dict_get(mstep_config, 'kernel', default='exp')
        
        # TF Graph stuff
        self.init_ = None # Variable initializer
        self.graph = None
        self.session, self.session_loss = None, None
        self.u, self.v, self.w, self.z, self.z0 = None, None, None, None, None # Representations
        self.A_from_reps_hmmlearn, self.B_from_reps_hmmlearn, self.pi_from_reps_hmmlearn = None, None, None # HMM parameters
        
        # Only needed for EM optimization
        self.em_epochs = dict_get(mstep_config, 'em_epochs', default=10)
        self.em_lr = dict_get(mstep_config, 'em_lr', default=0.1) # TODO: Actually only need em_optimizer
        self.em_optimizer = dict_get(mstep_config, 'em_optimizer', default=tf.train.GradientDescentOptimizer(self.em_lr))
        self.scaling = dict_get(mstep_config, 'scaling', default=n_hidden_states)
        self.gamma, self.bar_gamma, self.bar_gamma_pairwise = None, None, None # Placeholders
        self.tilde_O, self.tilde_O_ph = None, None # Input sequence
        self.loss_1, self.loss_1_normalization, self.loss_2, self.loss_2_normalization, self.loss_3, self.loss_3_normalization = None, None, None, None, None, None
        self.loss_scaled, self.loss_update = None, None # Loss to optimize
        
        # Only needed for cooc optimization
        self.cooc_lr = dict_get(mstep_config, 'cooc_lr', default=0.01) # TODO actually only need optimizer
        self.cooc_optimizer = dict_get(mstep_config, 'cooc_optimizer', default=tf.train.GradientDescentOptimizer(self.cooc_lr))
        self.cooc_epochs = dict_get(mstep_config, 'cooc_epochs', default=10000)
        self.loss_cooc, self.loss_cooc_update = None, None 
        self.A_stationary = None 
        self.omega, self.omega_gt_ph = None, None
        
    
    def _build_tf_em_graph(self, A_log_ker, B_log_ker, pi_log_ker, A_log_ker_normal, B_log_ker_normal, pi_log_ker_normal):
        
        with self.graph.as_default():
            
            # Placeholders
            gamma = tf.placeholder(name="gamma", dtype=tf.float64, 
                                   shape=[None, None, self.n_components])
            bar_gamma = tf.placeholder(name="bar_gamma", dtype=tf.float64,
                                       shape=[None, self.n_components])
            bar_gamma_pairwise = tf.placeholder(name="bar_gamma_pairwise", 
                                                dtype=tf.float64,
                                                shape=[None, self.n_components, self.n_components]) #TODO gamma or bar_gamma?
            tilde_O_ph = tf.placeholder(name="tilde_O", dtype=tf.float64,
                               shape=[None, None, self.n_observables])
            
            
            
            # M = B_log_ker
            # B = A_log_ker
            # B0 = pi_log_ker
            # S = B_log_ker_normal
            # L = A_log_ker_normal
            # L0 = pi_log_ker_normal
            
            # Losses  # TODO: Recheck this
            bar_gamma_1 = bar_gamma[0, :]
            loss_1 = -tf.reduce_sum(pi_log_ker * bar_gamma_1)
            loss_1_normalization = tf.reduce_sum(pi_log_ker_normal * bar_gamma_1)
            
            loss_2 = -tf.reduce_sum(A_log_ker * bar_gamma_pairwise)
            loss_2_normalization = tf.reduce_sum(
                A_log_ker_normal[tf.newaxis, :, tf.newaxis] * bar_gamma_pairwise)
            
            tilde_M = tf.einsum('ito,oh->ith', tilde_O_ph, B_log_ker)
            loss_3 = -tf.reduce_sum(gamma * tilde_M)
            loss_3_normalization = tf.reduce_sum(B_log_ker_normal * bar_gamma)
            
            loss_total = tf.identity(loss_1 + loss_1_normalization + 
                                     loss_2 + loss_2_normalization + 
                                     loss_3 + loss_3_normalization, 
                                     name="loss_total")
            loss_scaled = tf.identity(loss_total / self.scaling, name="loss_scaled")
            loss_1 = tf.identity(loss_1, name="loss_1")
            loss_1_normalization = tf.identity(loss_1_normalization,
                                               name="loss_1_normalization")
            loss_2 = tf.identity(loss_2, name="loss_2")
            loss_2_normalization = tf.identity(loss_2_normalization,
                                               name="loss_2_normalization")
            loss_3 = tf.identity(loss_3, name="loss_3")
            loss_3_normalization = tf.identity(loss_3_normalization,
                                               name="loss_3_normalization")
            
            # Optimizer step
            loss_update = self.em_optimizer.minimize(loss_scaled, name='loss_update')
            
            return gamma, bar_gamma, bar_gamma_pairwise, tilde_O_ph, loss_update, loss_scaled, loss_1, loss_1_normalization, loss_2, loss_2_normalization, loss_3, loss_3_normalization
    
    def _build_tf_coocs_graph(self, A_from_reps_hmmlearn, B_from_reps_hmmlearn, omega_gt):
        
        with self.graph.as_default():

            A = A_from_reps_hmmlearn
            B = B_from_reps_hmmlearn
            A_stationary = tf.placeholder(dtype=tf.float64, shape=[self.n_components]) # Assumed to be the eigenvector of A.T

            # Process of hidden variables is assumed to be stationary, pi being the stationary distribution
            # Then we get p(s_t = s_i, s_{t+1} = s_j) = p(s_{t+1} = s_j | s_t = s_i) sum_s p(s_t = s_i | s) p(s)
            # = A[i, j] pi[i]
            theta = A * A_stationary[:, None]  # theta[i, j] = p(s_t = s_i, s_{t+1} = s_j) = A[i, j] * pi[i]

            # omega[i, j] = P(O_{t} = o_i, O_{t+1} = o_j)
            # omega[i, j] = sum_{kl} B[k, i] theta[k, l] B[l, j] = sum_{kl} p(o_i | s_k) p(s_k, s_l) p(o_j | s_l)
            omega = tf.matmul(tf.transpose(B), tf.matmul(theta, B))
            loss_cooc = tf.reduce_sum(tf.square(omega_gt - omega))

            loss_cooc_update = self.cooc_optimizer.minimize(loss_cooc, var_list=[self.u, self.v, self.w, self.z])

            return loss_cooc, loss_cooc_update, A_stationary, omega
    
        
    def _build_tf_graph(self):
        
        
        if self.representations not in self.SUPPORTED_REPRESENTATIONS:
            raise Exception("Given representation argument is invalid. Has to be one of %s" % 
                            str(self.SUPPORTED_REPRESENTATIONS))

        if self.representations == 'vzz0' and self.l_vw != self.l_uz:
            raise Exception("Cannot use representation vzz0 while l_vw and l_uz differ")


        if len(self.opt_schemes.difference(self.SUPPORTED_OPT_SCHEMES)) > 0:
            raise Exception("Given unsupported optimization scheme! Supported are: %s" % str(self.SUPPORTED_OPT_SCHEMES))
        
        self.graph = tf.Graph()
        
        with self.graph.as_default():
            
            # TODO: does it work like that with the trainable = ...
            # Trainables
            v = tf.get_variable(name="v", dtype=tf.float64, shape=[self.n_observables, self.l_vw], 
                                initializer=self.initializer,
                                trainable=('v' in self.trainables))
            z = tf.get_variable(name="z", dtype=tf.float64, shape=[self.l_uz, self.n_components], 
                                initializer=self.initializer, 
                                trainable=('z' in self.trainables and 
                                           ('z0' not in self.trainables 
                                            or 'zz0' in self.trainables)))
            z0 = tf.get_variable(name="z0", dtype=tf.float64, shape=[self.l_uz, 1], 
                                initializer=self.initializer,
                                 trainable=('z0' in self.trainables))
            
            if self.representations == 'uvwzz0':
                u = tf.get_variable(name="u", dtype=tf.float64, shape=[self.n_components, self.l_uz], 
                                initializer=self.initializer, 
                                trainable=('u' in self.trainables))
            
                w = tf.get_variable(name="w", dtype=tf.float64, shape=[self.l_vw, self.n_components], 
                                initializer=self.initializer,
                                trainable=('w' in self.trainables))
            
            """ Recovering A, B, pi """
                   
            # Compute scalar products
            if self.representations == 'uvwzz0': # Convention here: B is m x n
                A_scalars = tf.matmul(u, z, name="A_scalars")
                B_scalars = tf.matmul(v, w, name="B_scalars")
                pi_scalars = tf.matmul(u, z0, name="pi_scalars")
                
            elif self.representations == 'vzz0':
                A_scalars = tf.matmul(z, z, transpose_a = True, name="A_scalars")
                B_scalars = tf.matmul(v, z, name="B_scalars")
                pi_scalars = tf.matmul(z0, z, transpose_a = True, name="pi_scalars")
            
            # Apply kernel
            if self.kernel == 'exp' or self.kernel == tf.exp:
                
                A_from_reps = tf.nn.softmax(A_scalars, axis=0)
                B_from_reps = tf.nn.softmax(B_scalars, axis=0)
                pi_from_reps = tf.nn.softmax(pi_scalars, axis=0)
                
                A_log_ker_normal = tf.reduce_logsumexp(A_scalars, axis=0) # L
                B_log_ker_normal = tf.reduce_logsumexp(B_scalars, axis=0) #Convention: Columns = Distribution for one state
                pi_log_ker_normal = tf.reduce_logsumexp(pi_scalars) #L0
                
                A_log_ker = tf.identity(A_scalars, name='A_log_ker')
                B_log_ker = tf.identity(B_scalars, name='B_log_ker')
                pi_log_ker = tf.identity(pi_scalars, name='pi_log_ker')
                
            else:
                A_scalars_ker = self.kernel(A_scalars)
                B_scalars_ker = self.kernel(B_scalars)
                pi_scalars_ker = self.kernel(pi_scalars)
                
                A_from_reps = A_scalars_ker / tf.reduce_sum(A_scalars_ker, axis=0)[tf.newaxis, :]
                B_from_reps = B_scalars_ker / tf.reduce_sum(B_scalars_ker, axis=0)[tf.newaxis, :]
                pi_from_reps = pi_scalars_ker / tf.reduce_sum(pi_scalars_ker)
                
                A_log_ker_normal = tf.log(tf.reduce_sum(A_scalars_ker, axis=0))
                B_log_ker_normal = tf.log(tf.reduce_sum(B_scalars_ker, axis=0))
                pi_log_ker_normal = tf.log(tf.reduce_sum(pi_scalars_ker))
                
                A_log_ker = tf.log(A_scalars_ker, name='A_log_ker')
                B_log_ker = tf.log(B_scalars_ker, name='B_log_ker')
                pi_log_ker = tf.log(pi_scalars_ker, name='pi_log_ker')
                                      
            # hmmlearn library uses a different convention for the shapes of the matrices
            A_from_reps_hmmlearn = tf.transpose(A_from_reps, name='A_from_reps')
            B_from_reps_hmmlearn = tf.transpose(B_from_reps, name='B_from_reps')
            pi_from_reps_hmmlearn = tf.reshape(pi_from_reps, (-1,), name='pi_from_reps')        
            
            # Member variables for convenience
            self.u, self.v, self.w, self.z, self.z0 = u, v, w, z, z0
            self.A_from_reps_hmmlearn, self.B_from_reps_hmmlearn, self.pi_from_reps_hmmlearn = A_from_reps_hmmlearn, B_from_reps_hmmlearn, pi_from_reps_hmmlearn
            
            # Build optimization graphs
            if 'em' in self.opt_schemes:
                self.gamma, self.bar_gamma, self.bar_gamma_pairwise, self.tilde_O_ph, self.loss_update, self.loss_scaled, self.loss_1, self.loss_1_normalization, self.loss_2, self.loss_2_normalization, self.loss_3, self.loss_3_normalization = self._build_tf_em_graph(A_log_ker, B_log_ker, pi_log_ker, A_log_ker_normal, B_log_ker_normal, pi_log_ker_normal)
            
            if 'cooc' in self.opt_schemes:
                self.omega_gt_ph = tf.placeholder(dtype=tf.float64, shape=[self.n_observables, self.n_observables])
                self.loss_cooc, self.loss_cooc_update, self.A_stationary, self.omega = self._build_tf_coocs_graph(A_from_reps_hmmlearn, B_from_reps_hmmlearn, self.omega_gt_ph)
            
            self.init_ = tf.global_variables_initializer()
    
    def _init_tf(self):
        self._build_tf_graph()
        self.session = tf.Session(graph=self.graph)
        self.session.run(self.init_)
        self.startprob_ = self.session.run(self.pi_from_reps_hmmlearn)
        self.transmat_ = self.session.run(self.A_from_reps_hmmlearn)
        self.emissionprob_ = self.session.run(self.B_from_reps_hmmlearn)
                     
    def _init(self, X, lengths=None):
        X, n_seqs, max_seqlen = super(DenseHMM, self)._init(X, lengths=lengths)
        
        if 'em' in self.opt_schemes:
            O, n_seqs, max_seqlen = self._observations_to_padded_matrix(X, lengths)
            self.tilde_O = np.eye(self.n_observables)[O]
        
        self._init_tf()
        return X, n_seqs, max_seqlen
        
    
    """ Learns representations, recovers transition matrices and sets them """
    def _do_mstep(self, stats):
        
        if self.session is None:
            raise Exception("Uninitialized TF Session. You must call _init first")
            
        for epoch in range(self.em_epochs):
            train_input_dict = {self.gamma: stats['gamma'], 
                                self.bar_gamma: stats['bar_gamma'],
                                self.bar_gamma_pairwise: stats['bar_gamma_pairwise'],
                                self.tilde_O_ph: self.tilde_O}
            self.session.run(self.loss_update, feed_dict=train_input_dict)
            print("Loss at epoch %d is %.8f" % (epoch, self.session.run(self.loss_scaled, feed_dict=train_input_dict)))
            
        A, B, pi = self.session.run([self.A_from_reps_hmmlearn, self.B_from_reps_hmmlearn, self.pi_from_reps_hmmlearn])
        self.transmat_ = A
        self.emissionprob_ = B
        self.startprob_ = pi
        
        #print(A, B, pi)
        
        
    def _compute_metrics(self, X, lengths, stats, em_iter, ident, 
                         val=None, val_lengths=None):
        
        
        log_config = self.logging_monitor.log_config
        log_dict = super(DenseHMM, self)._compute_metrics(X, lengths, stats, em_iter, ident, val, val_lengths)
        
        log_dict['u'], log_dict['v'], log_dict['w'], log_dict['z'], log_dict['z_0'] = self.get_representations()
        
        gamma, bar_gamma = stats['gamma'], stats['bar_gamma']
        bar_gamma_pairwise = stats['bar_gamma_pairwise']
        log_dict['train_losses'] = self._compute_loss(X, lengths, bar_gamma, bar_gamma_pairwise, gamma)
        log_dict['train_losses_standard'] = self._compute_loss_standard(X, lengths, bar_gamma, bar_gamma_pairwise, gamma)
        
        if val is not None:
            
            log_dict['test_losses'] = self._compute_loss(val, val_lengths, bar_gamma, bar_gamma_pairwise, gamma)
            log_dict['test_losses_standard'] = self._compute_loss_standard(val, val_lengths, bar_gamma, bar_gamma_pairwise, gamma)
            
            # val-gammas are not necessarily in log_dict
            val_bar_gamma = dict_get(log_dict, 'val_bar_gamma')
            val_bar_gamma_pairwise = dict_get(log_dict, 'val_bar_gamma_pairwise')
            val_gamma = dict_get(log_dict, 'val_gamma')
            
            if val_bar_gamma is not None and val_bar_gamma_pairwise is not None and val_gamma is not None:
                log_dict['test_gamma_losses'] = self._compute_loss(val, val_lengths, val_bar_gamma, val_bar_gamma_pairwise, val_gamma)
                log_dict['test_gamma_losses_standard'] = self._compute_loss_standard(val, val_lengths, val_bar_gamma, val_bar_gamma_pairwise, val_gamma) 
    
        return log_dict
    
    """ Computes same loss as standard hmm """
    def _compute_loss_standard(self, X, lengths, bar_gamma, bar_gamma_pairwise, gamma):
        
        X, n_seqs, max_seqlen = self._observations_to_padded_matrix(X, lengths)
        
        log_A = np.log(self.transmat_)
        log_B = np.log(self.emissionprob_)
        log_pi = np.log(self.startprob_)
        
        tilde_B = log_B[:, X] # Has shape (n_hidden_states, seqs, max_seqlen)
        
        loss1 = -np.einsum('s,s->', log_pi, bar_gamma[0, :])
        loss2 = -np.einsum('jl,tjl->', log_A, bar_gamma_pairwise)
        loss3 = -np.einsum('jit,itj->', tilde_B, gamma) 
        loss = loss1 + loss2 + loss3
        
        return np.array([loss, loss1, loss2, loss3])
            
        
    def _compute_loss(self, X, lengths, bar_gamma, bar_gamma_pairwise, gamma):
        
        O, n_seqs, max_seqlen = self._observations_to_padded_matrix(X, lengths)
        tilde_O = np.eye(self.n_observables)[O]
        
        input_dict = {self.gamma: gamma, 
                      self.bar_gamma: bar_gamma,
                      self.bar_gamma_pairwise: bar_gamma_pairwise,
                      self.tilde_O_ph: tilde_O
                     }
        losses = self.session.run([self.loss_1, self.loss_1_normalization, 
                               self.loss_2, self.loss_2_normalization, 
                               self.loss_3, self.loss_3_normalization], feed_dict=input_dict)
        losses = [np.sum(losses)] + losses
        return np.array(losses)
        
   
    """ Fits a DenseHMM using the co-occurrence optimization scheme 
        If gt_AB = (A, B) is given, X/val are assummed to be generated by a
        stationary HMM with parameters A, B and gt co-occurence is computed analytically
        
    """
    def fit_coocs(self, X, lengths, val=None, val_lengths=None, gt_AB=None):
        
        X, n_seqs, max_seqlen = self._init(X, lengths)
        
        gt_omega = None
        freqs, gt_omega_emp = empirical_coocs(X, self.n_observables, lengths=lengths)
        gt_omega_emp = np.reshape(gt_omega_emp, newshape=(self.n_observables, self.n_observables))
        
        if gt_AB is not None:
            A, B = gt_AB
            A_stationary = compute_stationary(A)
            theta = A * A_stationary[:, None]
            gt_omega = np.matmul(B.T, np.matmul(theta, B))
        
        gt_omega = gt_omega_emp if gt_omega is None else gt_omega
        log_dict = self._fit_coocs(gt_omega)
                                             
        log_dict['cooc_logprobs'] = self.score_individual_sequences(X, lengths)[0]
        if val is not None and val_lengths is not None:
            log_dict['cooc_val_logprobs'] = self.score_individual_sequences(val, val_lengths)[0]
            
        self.logging_monitor.log('logs_coocs', log_dict)
                                             
    def _fit_coocs(self, omega_gt):
         
        if self.session is None:
            raise Exception("Unintialized session")
        
        A_, B_ = self.A_from_reps_hmmlearn, self.B_from_reps_hmmlearn
        A_stationary_ = self.A_stationary
        omega_ = self.omega
        
        def get_ABA_stationary():
            A, B = self.session.run([A_, B_])
            
            # TODO: As Tf v1 does not support eigenvector computation for
            # non-symmetric matrices, need to do this with numpy and feed
            # the result into the graph
            return A, B, compute_stationary(A, verbose=False) 
          
        feed_dict = {self.omega_gt_ph: omega_gt, A_stationary_: None}
        losses = []
        
        for epoch in range(self.cooc_epochs):
            
            A, B, A_stationary = get_ABA_stationary()
            feed_dict[A_stationary_] = A_stationary
            
            self.session.run(self.loss_cooc_update, feed_dict=feed_dict)
            cur_loss = self.session.run(self.loss_cooc, feed_dict=feed_dict)
            losses.append(cur_loss)
                     
            if epoch % 1000 == 0:
                print(cur_loss)
                     
        log_dict = {}
        log_dict['cooc_losses'] = losses
                     
        A, B, A_stationary = get_ABA_stationary()
        feed_dict[A_stationary_] = A_stationary
        learned_omega = self.session.run(self.omega, feed_dict)
                     
        self.transmat_ = A
        self.emissionprob_ = B
        self.startprob_ = A_stationary
        self._check()
                     
        log_dict.update({'cooc_transmat': self.transmat_, 'cooc_emissionprob': self.emissionprob_, 'cooc_startprob': self.startprob_, 'cooc_omega': learned_omega})
        
        u, v, w, z = self.session.run([self.u, self.v, self.w, self.z])
        log_dict.update(dict(u=u, v=v, w=w, z=z))
                     
        if self.logging_monitor.log_config['samples_after_cooc_opt'] is not None:
            sample_sizes = None
            if type(self.logging_monitor.log_config['samples_after_cooc_opt']) == tuple:
                sample_sizes = self.logging_monitor.log_config['samples_after_cooc_opt']
            else:
                if val_lengths is not None:
                    sample_sizes = (len(val_lengths), np.max(val_lengths))
                else:
                    sample_sizes = (len(lengths), np.max(lengths))
            log_dict['cooc_samples'] = self.sample_sequences(*sample_sizes)
                
        return log_dict
    
    def get_representations(self):
        return self.session.run([self.u, self.v, self.w, self.z, self.z0])
        
        
    