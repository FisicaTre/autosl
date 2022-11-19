from gwpy.table import EventTable
from gwpy.time import to_gps
import pandas as pd
import numpy as np
from gwpy.timeseries import TimeSeriesDict
import pytvfemd
import os
from scipy.signal import hilbert
from scipy.stats import pearsonr
from scipy.signal import lfilter, butter


SL_DATA = "./data/sl"
NO_SL_DATA = "./data/no_sl"
LAMBDA = 1.064
PREDICTOR = "SUS-ETMX_L2_WIT_L_DQ"
FS = 100
INTERVAL = 30
MAX_IMFS = 10
SMOOTH_WIN = 50
CORR_THR = 0.7
LOWPASS = 80


def get_glitches(gps1, gps2, save_path=None):
    glitches_list = EventTable.fetch("gravityspy", "glitches_v2d0",
                                     selection=[ # "ml_label=Scattered_Light",
                                                "0.9<=ml_confidence<=1.0",
                                                "10<=snr<=20",
                                                "ifo=L1",
                                                "{}<event_time<{}".format(gps1, gps2)],
                                     host="gravityspyplus.ciera.northwestern.edu",
                                     user="mla", passwd="gl1tch35Rb4d!")

    glitches_list = glitches_list.to_pandas()
    glitches_list.drop_duplicates("peak_time", keep=False, inplace=True)

    if save_path is not None:
        glitches_list.to_csv(save_path, index=False)

    return glitches_list


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


def get_predictor(channel, fs, smooth_win=None, n_scattering=1):
    time = np.arange(0, len(channel) / fs, 1 / fs, dtype=float)
    v_mat = np.diff(channel) / np.diff(time)
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
    t1, t2 = to_gps("2019-04-01"), to_gps("2020-03-27")
    glitches = pd.read_csv("./glitches.csv")
    # glitches = get_glitches(t1, t2, "./glitches.csv")

    for i, g in glitches.iterrows():
        channels_list = [ifo_and_channel(g.ifo, g.channel)]
        if g.ml_label == "Scattered_Light":
            channels_list.append(ifo_and_channel(g.ifo, PREDICTOR))
        data_dict = TimeSeriesDict.get(channels_list, g.peak_time - INTERVAL, g.peak_time + INTERVAL)
        data_dict.resample(FS)

        filtered_channel = butter_lowpass_filter(data_dict[channels_list[0]].value, LOWPASS, FS)

        imfs = pytvfemd.tvfemd(filtered_channel, max_imf=MAX_IMFS + 1)
        imfs = (imfs - np.nanmean(imfs, axis=0)) / np.nanstd(imfs, axis=0)
        for nimf in imfs.shape[1]:
            file_name = "t{:d}_fs{:d}_imf{:d}".format(g.peak_time, FS, nimf + 1)
            if g.ml_label == "Scattered_Light":
                predictor = get_predictor(data_dict[channels_list[1]].value, FS, SMOOTH_WIN)
                upper_env = upper_envelope(imfs[:, nimf])[1:]
                upper_env = smooth(upper_env, SMOOTH_WIN)
                corr = get_correlation_between(predictor, upper_env)
                if np.isnan(corr) or corr < CORR_THR:
                    np.savetxt(os.path.join(NO_SL_DATA, file_name), imfs[:, nimf])
                else:
                    np.savetxt(os.path.join(SL_DATA, file_name), imfs[:, nimf])
            else:
                np.savetxt(os.path.join(NO_SL_DATA, file_name), imfs[:, nimf])
        break
