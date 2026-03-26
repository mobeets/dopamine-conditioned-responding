#%% imports

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import warnings
# from statsmodels.tools.sm_exceptions import ConvergenceWarning
import pandas as pd

#%% helper functions

def zscore_if_possible(y, mu=None, sd=None):
    changed_shape = False
    if len(y.shape) == 1:
        changed_shape = True
        y = y[:,None]
        mu = mu[:,None] if mu is not None else mu
        sd = sd[:,None] if sd is not None else sd
    sds = np.nanstd(y, axis=0) if sd is None else sd
    sds[sds == 0] = 1 # if no sdev, we won't divide by it
    mu = np.nanmean(y, axis=0) if mu is None else mu
    z = (y - mu)/sds
    return z[:,0] if changed_shape else z

#%% functions

def fit_linear_mixed_effects_model(data, alpha=0, ignore_X=False, nfolds=5, seed=666):
    """
    - data (dict):
        - X is [Tx1] input data
        - y is [Tx1] observations
        - S is [Tx1] grouping variables (with K distinct entries)
    """
    # Convert to DataFrame
    X = data['X']
    X = (X - np.mean(X)) / np.std(X) # z-score for stability
    S = data['S']
    S = S.astype(str) # Convert S to string for categorical handling
    y = data['y']
    import matplotlib.pyplot as plt
    # plt.scatter(X, y, 2, c=S.astype(int)), plt.show()
    Sc = S.astype(int)
    plt.figure(figsize=(10,10))
    for c in np.unique(Sc):
        plt.subplot(5,5,c+1)
        plt.scatter(X[Sc == c], y[Sc == c], 2)
        print(c, np.corrcoef([X[Sc==c], y[Sc==c]])[0,1])
    plt.tight_layout()
    plt.show()
    data = {'X': X, 'y': y, 'S': S}
    return

    # Fit linear mixed model: fixed effect for X, random intercepts & slopes by S
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)

    scores = []
    for train_idx, test_idx in kf.split(data['X']):
        X_train, X_test = data['X'][train_idx], data['X'][test_idx]
        y_train, y_test = data['y'][train_idx], data['y'][test_idx]
        train_data = pd.DataFrame({'X': X_train, 'y': y_train, 'S': data['S'][train_idx]})
        test_data = pd.DataFrame({'X': X_test, 'y': y_test, 'S': data['S'][test_idx]})

        if ignore_X:
            model = smf.mixedlm("y ~ 1", train_data, groups=train_data["S"], re_formula="1")
            print(train_data.groupby('S')['y'].mean())
            print(train_data.groupby('S')['y'].var())
        else:
            model = smf.mixedlm("y ~ X", train_data, groups=train_data["S"], re_formula="1 + X")
        with warnings.catch_warnings():
            # warnings.simplefilter("ignore", category=ConvergenceWarning)
            if alpha == 0:
                result = model.fit(method='lbfgs', maxiter=10000)
            else:
                result = model.fit_regularized(alpha=alpha)
        
        # get full predictions on test data
        print(result.summary())
        print(result.cov_re)
        random_effects = result.random_effects # dict mapping group -> random intercept/slope
        random_effects_arr = np.vstack([random_effects[s] for s in test_data['S']])
        random_intercepts = random_effects_arr[:,0]
        if ignore_X:
            yhat_test = result.predict(test_data) + random_intercepts
        else:
            random_slopes = random_effects_arr[:,1]
            yhat_test = result.predict(test_data) + random_intercepts + (random_slopes * test_data['X'])

        rsq = 1 - np.var(y_test-yhat_test)/np.var(y_test)
        scores.append(rsq)  # R² score

    rsq_mu = np.mean(scores)
    rsq_se = np.std(scores)/np.sqrt(len(scores))
    return {'model': model, 'result': result, 'data': data, 'rsq_mu': rsq_mu, 'rsq_se': rsq_se, 'ntrials': len(y)}

def cv_scores(data, model, nfolds=5, seed=666):
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in kf.split(data['X']):
        X_train, X_test = data['X'][train_idx], data['X'][test_idx]
        y_train, y_test = data['y'][train_idx], data['y'][test_idx]
        
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))  # R² score
    return scores

def get_feature_from_trials(trials, field_name, window=None, trial_types=None, mode='mean', trial_type_key='trial_type'):
    trial_keeper = lambda trial: (trial_types is None) or (trial[trial_type_key] in trial_types)

    data = [trial[field_name] for trial in trials if trial_keeper(trial)]
    if len(data) == 0:
        return np.array([])
    else:
        data = np.vstack(data)
    if window is not None:
        data = data[:,window[0]:window[1]]
    if mode == 'mean':
        X = data.mean(axis=1)
    elif mode == 'max':
        X = data.max(axis=1)
    elif mode == 'nanmean':
        X = np.nanmean(data, axis=1)
    else:
        raise Exception(f"Unrecognized {mode=}. Options are 'max' or 'mean'.")
    return X

def add_random_slopes(X, S):
    """
    X is [Tx1]; S is [Tx1], with K discrete entries
    returns [TxK], where X[t,k] = X[t] if S[t]==k
    """
    K = np.unique(S).max()
    X_new = np.zeros((len(X), K+1))
    for k in range(K+1):
        X_new[S==k,k] = X[S==k]
    return X_new

def get_data_per_session(session, da_window, lick_window, trial_types=None, zscore_spikes=False, use_mean_spikes=True, subtract_baseline_spikes=False, trial_type_key='trial_type'):
    # collect DA, licking, and session index data
    X = get_feature_from_trials(session, 'DA', da_window, trial_types, 'mean' if use_mean_spikes else 'max', trial_type_key=trial_type_key)
    if subtract_baseline_spikes:
        Xb = get_feature_from_trials(session, 'baseline_DA_rate', None, trial_types, 'mean', trial_type_key=trial_type_key)
        # Xb is per bin, just like X
        X = X - Xb # *(da_window[1]-da_window[0])
    Y = get_feature_from_trials(session, 'licks', lick_window, trial_types, 'mean', trial_type_key=trial_type_key)
    X = zscore_if_possible(X) if zscore_spikes else X
    return X, Y

def fit_linear_model_per_animal(Sessions, da_window, lick_window, key_fields=('mouse',), trial_types=None, day_indices=None, nfolds=10, use_mean_spikes=True, subtract_baseline_spikes=False, alpha=0.0, bin_size=50, reduce_matching_sessions=False, day_shift=0):

    datas = {}
    n_matches = 0
    for i, session in enumerate(Sessions):
        if day_indices is not None and session[0]['day_index'] not in day_indices:
            continue
        # get DA and licking
        X, Y = get_data_per_session(session, da_window, lick_window, trial_types=trial_types, zscore_spikes=False, use_mean_spikes=use_mean_spikes, subtract_baseline_spikes=subtract_baseline_spikes)
        session_name = tuple(session[0][key_field] for key_field in key_fields)

        if day_shift > 0:
            X = X[day_shift:]
            Y = Y[:-day_shift]
        elif day_shift < 0:
            X = X[:day_shift]
            Y = Y[-day_shift:]

        if session_name not in datas:
            datas[session_name] = {'X': [X], 'y': [Y], 'n': 1, 'len': len(Y)}
        elif reduce_matching_sessions:
            n = datas[session_name]['n']
            if datas[session_name]['len'] == len(Y) and np.isclose(datas[session_name]['y'], Y).all(): # matching session
                X = (n*datas[session_name]['X'][0] + X) / (n+1)
            else:
                raise Exception('WARNING: matching session keys but not matching licks')
            datas[session_name]['X'] = [X]
            datas[session_name]['n'] = n+1
            n_matches += 1
        else:
            datas[session_name]['X'].append(X)
            datas[session_name]['y'].append(Y)
    if n_matches > 0:
        print(f'Combined {n_matches} matching sessions.')
    
    results = []
    for session_name, data in datas.items():
        X = np.hstack(data['X']).reshape(-1,1)
        Y = np.hstack(data['y'])

        model = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
        scores = cv_scores({'X': zscore_if_possible(X), 'y': Y}, model, nfolds=nfolds)
        mu = np.mean(scores)
        se = np.std(scores)/np.sqrt(len(scores))
        result = {'name': session_name, 'model': model, 'X': X, 'y': Y, 'rsq_mu': mu, 'rsq_se': se, 'nfolds': nfolds, 'ntrials': len(Y)}
        results.append(result)
    return results

def fit_linear_models_combined(Sessions, da_window, lick_window, key_fields, trial_types=None, day_indices=None, nfolds=10, zscore_spikes=False, use_mean_spikes=True, reduce_matching_sessions=True, subtract_baseline_spikes=False, alpha=0.0, bin_size=50):
    
    # collect DA, licking, and session index data per session
    Xs = []; Ys = []; Ss = []
    key_inds_found = []
    key_index_lookup = {}; last_key_index = 0
    for i, session in enumerate(Sessions):
        if day_indices is not None and session[0]['day_index'] not in day_indices:
            continue
        # get DA and licking
        X, Y = get_data_per_session(session, da_window, lick_window, trial_types=trial_types, zscore_spikes=zscore_spikes, use_mean_spikes=use_mean_spikes, subtract_baseline_spikes=subtract_baseline_spikes)

        # get session index
        key = tuple([str(session[0][key_field]) for key_field in key_fields])
        if key in key_index_lookup:
            key_index = key_index_lookup[key]
        else:
            key_index = last_key_index
            key_index_lookup[key] = key_index
            last_key_index += 1
        S = key_index*np.ones_like(Y)
        key_inds_found.append(key_index)
        Xs.append(X); Ys.append(Y); Ss.append(S)

    # identify duplicate sessions based on key_names_found
    n_duplicate_sessions = len(key_inds_found) - len(np.unique(key_inds_found))
    if n_duplicate_sessions > 0:
        if reduce_matching_sessions:
            print(f'Reducing {n_duplicate_sessions} duplicate sessions.')
            knms = np.unique(key_inds_found)
            n = len(key_inds_found) - len(knms)
            Xs2 = []; Ys2 = []; Ss2 = []
            for knm in knms:
                inds = np.array([i for i,cnm in enumerate(key_inds_found) if knm==cnm])
                Xsc = np.vstack([Xs[i] for i in inds]).mean(axis=0)
                Ysc = np.vstack([Ys[i] for i in inds]).mean(axis=0)
                Ssc = np.vstack([Ss[i] for i in inds]).mean(axis=0)
                Xs2.append(Xsc); Ys2.append(Ysc); Ss2.append(Ssc)
            Xs = Xs2; Ys = Ys2; Ss = Ss2
        else:
            print(f'WARNING: {n_duplicate_sessions} duplicate sessions exist.')

    Xs = np.hstack(Xs); Y = np.hstack(Ys); Ss = np.hstack(Ss).astype(int)
    Xrs = add_random_slopes(Xs, Ss)
    S = np.eye(Ss.max()+1)[Ss] # one-hot encoding of mouse name

    # fit models
    X0 = np.hstack([Xrs, Xs.reshape(-1,1), S]); nm0 = 'full'
    X1 = np.hstack([Xs.reshape(-1,1), S]); nm1 = 'fixed slope, fixed/random intercepts'
    X2 = np.hstack([Xrs, Xs.reshape(-1,1)]); nm2 = 'fixed/random slopes, fixed intercept'
    X3 = np.hstack([S]); nm3 = 'fixed/random intercepts'
    X4 = np.hstack([Xs.reshape(-1,1)]); nm4 = 'fixed slope, fixed intercept'

    results = {}
    for name,X in {nm0: X0, nm1: X1, nm2: X2, nm3: X3, nm4: X4}.items():
        sds = np.std(X, axis=0)
        sds[sds <= 0] = 1.
        X = (X - np.mean(X, axis=0)) / sds # z-score for stability

        model = LinearRegression() if alpha == 0 else Ridge(alpha=alpha)
        scores = cv_scores({'X': X, 'y': Y}, model, nfolds=nfolds)
        mu = np.mean(scores)
        se = np.std(scores)/np.sqrt(len(scores))
        results[name] = {'model': model, 'X': X, 'y': Y, 'rsq_mu': mu, 'rsq_se': se, 'nfolds': nfolds, 'ntrials': len(Y), 'key_fields': key_fields, 'key_index_lookup': key_index_lookup}
    
    results['statsmodels, no DA'] = fit_linear_mixed_effects_model({'X': Xs, 'y': Y, 'S': Ss}, alpha=0, nfolds=nfolds, ignore_X=True)
    results['statsmodels'] = fit_linear_mixed_effects_model({'X': Xs, 'y': Y, 'S': Ss}, alpha=0, nfolds=nfolds)
    return results

#%% predicting probability of licking

def logistic_regression_cv(X, y, nfolds=10, train_test_indices=None):
    """
    Perform 10-fold cross-validation on Logistic Regression and return d-prime and AUC.
    """
    if train_test_indices is None:
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=42)
        train_test_indices = kf.split(X)
    d_prime_scores = []
    auc_scores = []
    
    for train_idx, test_idx in train_test_indices:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Add intercept
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)
        
        # Fit Logistic Regression
        model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
        result = model.fit()
        
        # Predict on test set
        y_pred = result.predict(X_test)
        
        # Compute AUC
        auc = roc_auc_score(y_test, y_pred)
        auc_scores.append(auc)
        
        # Compute d-prime
        hit_rate = np.mean(y_pred[y_test == 1])
        false_alarm_rate = np.mean(y_pred[y_test == 0])
        
        # Avoid infinite z-scores
        hit_rate = np.clip(hit_rate, 1e-5, 1 - 1e-5)
        false_alarm_rate = np.clip(false_alarm_rate, 1e-5, 1 - 1e-5)

        d_prime = norm.ppf(hit_rate) - norm.ppf(false_alarm_rate)
        d_prime_scores.append(d_prime)
    
    if len(auc_scores) > 1: # fit model to all data
        Xc = sm.add_constant(X)
        model = sm.GLM(y, Xc, family=sm.families.Binomial())
        result = model.fit()
    
    X_range = np.linspace(X.min(), X.max())
    Xc = sm.add_constant(X_range)
    yhat = result.predict(Xc)

    return {
        'd_prime_mean': np.mean(d_prime_scores),
        'auc_mean': np.mean(auc_scores),
        'd_prime_scores': d_prime_scores,
        'auc_scores': auc_scores,
        'X_range': X_range,
        'yhat': yhat,
    }

def fit_logistic_models(D, ntrials_per_chunk=10, split_by_trial_type=False):
    fcue = lambda ttype: ['go', 'no', 'c', 'background', 'unpred'].index(ttype.split('_')[0])

    pts = {}
    scores = {}
    for mouse_name in D.mouse_names:
        xs = []
        trial_counts = {} # track num. trials per odor so far
        for trials in D.Sessions:
            for trial in trials:
                if trial['mouse'] != mouse_name:
                    continue
                if trial['trial_type'] in ['background', 'unpred', 'unpred_water']:
                    continue
                if trial['day_index'] > 12:
                    continue

                cue_index = fcue(trial['trial_type'])
                if cue_index not in trial_counts:
                    trial_counts[cue_index] = 0
                else:
                    trial_counts[cue_index] += 1
                window_index = trial_counts[cue_index] // ntrials_per_chunk
                x = (trial['day_index'], window_index, cue_index, trial['CS_DA_peak'], trial['did_anticipatory_lick'])
                xs.append(x)
        xs = np.vstack(xs)
        pts[mouse_name] = xs

        if split_by_trial_type:
            # split by trial type
            ixTrain = (xs[:,2] == 0) # go
            train_test_indices = ((ixTrain, ~ixTrain),)
            scores[mouse_name] = logistic_regression_cv(xs[:,-2], xs[:,-1], train_test_indices=train_test_indices)
        else:
            scores[mouse_name] = logistic_regression_cv(xs[:,-2], xs[:,-1])

    return pts, scores
