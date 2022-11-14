from gwpy.table import EventTable
from gwpy.time import to_gps
import pandas as pd
import numpy as np
from gwpy.timeseries import TimeSeriesDict
import pytvfemd


PREDICTOR = "SUS-ETMX_L2_WIT_L_DQ"
FS = 100
INTERVAL = 30
MAX_IMFS = 10


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
        # filter ?
        imfs = pytvfemd.tvfemd(channels_list[0], max_imf=MAX_IMFS + 1)
        imfs = (imfs - np.nanmean(imfs, axis=0)) / np.nanstd(imfs, axis=0)
        n_imfs = imfs.shape[1]
