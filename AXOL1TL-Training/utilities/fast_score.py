import numpy as np


def fast_score(model, data_bkg, bkg_l1_bits, distance_func, data_signal, signal_l1_bits, evaluation_threshold, tf_backend = False):
    
    if tf_backend:
        score_background = distance_func(data_bkg)
    else:
        score_background = distance_func(model=model, data=data_bkg)
        
    threshold = {}
    target_rate = [0.15, 0.2, 0.3, 0.6, 1, 2, 3, 5, 10]
    rates = [0.15, 0.2, 0.3, 0.6, 1, 2, 3, 5, 10]
    bc_rate_khz = 11245.6 * 2.544
    for rate in rates:
        threshold[rate] = np.percentile(score_background, 100 - (rate / bc_rate_khz) * 100)

    raw_rate = []
    pure_rate = []
    for thres in threshold.keys():
        nsamples = score_background.shape[0]
        axo_triggered = np.where(score_background > threshold[thres])[0]
        l1_triggered = np.where(bkg_l1_bits)[0]
        pure_triggered = np.setdiff1d(axo_triggered, l1_triggered)
        raw_rate.append((axo_triggered.shape[0] * bc_rate_khz) / nsamples)
        pure_rate.append((pure_triggered.shape[0] * bc_rate_khz) / nsamples)

    threshold_pure = {}
    for thres in target_rate:
        _pure_rate = thres
        _raw_rate = np.interp(_pure_rate, xp=pure_rate, fp=raw_rate)
        threshold_pure[thres] = np.percentile(score_background, 100 - (_raw_rate / bc_rate_khz) * 100)

    score = {}

    _raw = []
    _pure = []
    _l1 = []
    SIGNAL_NAMES = list(data_signal.keys())

    for signal_name in SIGNAL_NAMES:
        if tf_backend:
            score_signal = distance_func(data_signal[signal_name])
        else:
            score_signal = distance_func(model, data_signal[signal_name])

        nsamples = score_signal.shape[0]

        raw_triggered = np.where(score_signal > threshold_pure[evaluation_threshold])[0]
        pure_triggered = np.setdiff1d(ar1=raw_triggered, ar2=np.where(signal_l1_bits[signal_name])[0])

        raw_rate = raw_triggered.shape[0] / nsamples
        pure_rate = pure_triggered.shape[0] / nsamples

        score[f'raw-pure/{signal_name}'] = raw_rate * 100
        score[f'pure-pure/{signal_name}'] = pure_rate * 100

    return score
