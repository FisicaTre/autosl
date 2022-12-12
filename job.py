#!/home/stefano.bianchi/.conda/envs/scattering_env/bin/python3

from gwpy.timeseries import TimeSeriesDict
import pytvfemd
import os
import numpy as np
from scipy.signal import lfilter, butter
from scipy.signal import hilbert
from scipy.stats import pearsonr
import argparse


SL_DATA = os.path.join("data", "sl")
NO_SL_DATA = os.path.join("data", "no_sl")
LAMBDA = 1.064
PREDICTOR = "SUS-ETMX_L2_WIT_L_DQ"
FS = 100
INTERVAL = 31
MAX_IMFS = 10
SMOOTH_WIN = 50
CORR_THR = 0.7


def ifo_and_channel(s_ifo, s_channel):
    return s_ifo + ":" + s_channel


def butter_lowpass(cutoff, f_samp, order=3):
    nyq = 0.5 * f_samp
    normal_cutoff = cutoff / nyq
    response = butter(order, normal_cutoff, btype="lowpass", output="ba", analog=False)

    return response[0], response[1]


def butter_lowpass_filter(x, cutoff, f_samp, order=3):
    b, a = butter_lowpass(cutoff, f_samp, order=order)
    y = lfilter(b, a, x)

    return y


def smooth(arr, win):
    if win % 2 == 0:
        win += 1
    out = np.convolve(arr, np.ones(win, dtype=int), "valid") / win
    r = np.arange(1, win - 1, 2)
    start = np.cumsum(arr[:win - 1])[::2] / r
    stop = (np.cumsum(arr[:-win:-1])[::2] / r)[::-1]

    return np.concatenate((start, out, stop))


def get_predictor(chnl, fs, smooth_win=None, n_scattering=1):
    time = np.arange(0, len(chnl) / fs, 1 / fs, dtype=float)
    v_mat = np.diff(chnl) / np.diff(time)
    if smooth_win is not None:
        v_mat = smooth(v_mat, smooth_win)
    pred = n_scattering * (2 / LAMBDA) * np.abs(v_mat)

    return pred


def upper_envelope(ts):
    analytic_signal = hilbert(ts)
    upp_env = np.abs(analytic_signal)

    return upp_env


def get_correlation_between(x, y):
    r_corr, _ = pearsonr(x, y)

    return r_corr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ifo", required=True, help="interferometer")
    ap.add_argument("--channel", required=True, help="channel")
    ap.add_argument("--ml_label", required=True, help="ML label")
    ap.add_argument("--peak_time", required=True, type=int, help="glitch time")
    ap.add_argument("--peak_freq", required=True, help="glitch peak frequency")
    ap.add_argument("--out_path", required=True, help="absolute path of the output results")
    args = vars(ap.parse_args())

    ifo = args["ifo"]
    channel = args["channel"]
    ml_label = args["ml_label"]
    peak_time = args["peak_time"]
    peak_freq = args["peak_freq"]
    out_path = args["out_path"]

    channels_list = [ifo_and_channel(ifo, channel)]
    if ml_label == "Scattered_Light":
        channels_list.append(ifo_and_channel(ifo, PREDICTOR))
    data_dict = TimeSeriesDict.get(channels_list, peak_time - INTERVAL, peak_time + INTERVAL)
    data_dict.resample(FS)

    filtered_channel = butter_lowpass_filter(data_dict[channels_list[0]].value, peak_freq, FS)

    imfs = pytvfemd.tvfemd(filtered_channel, max_imf=MAX_IMFS + 1)
    imfs = (imfs - np.nanmean(imfs, axis=0)) / np.nanstd(imfs, axis=0)
    for nimf in imfs.shape[1]:
        file_name = "t{:d}_fs{:d}_imf{:d}.dat".format(peak_time, FS, nimf + 1)
        upper_env = upper_envelope(imfs[:, nimf])[1:]
        ia = smooth(upper_env[FS:-FS], SMOOTH_WIN)
        if ml_label == "Scattered_Light":
            predictor = get_predictor(data_dict[channels_list[1]].value, FS, SMOOTH_WIN)[FS:-FS]
            corr = get_correlation_between(predictor, ia)
            if np.isnan(corr) or corr < CORR_THR:
                np.save(os.path.join(out_path, NO_SL_DATA, file_name), ia)
            else:
                np.save(os.path.join(out_path, SL_DATA, file_name), ia)
        else:
            np.save(os.path.join(out_path, NO_SL_DATA, file_name), ia)
