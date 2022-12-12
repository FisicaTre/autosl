from gwpy.table import EventTable
from gwpy.time import to_gps
import pandas as pd
import os
from gwas_tools.helpers import sub_file
from gwas_tools.helpers import dag_file


ANALYSIS_PATH = os.path.expandvars("$HOME/analyses/autosl")
ACCOUNTING_GROUP = "ligo.prod.o3.detchar.explore.test"


def get_glitches(gps1, gps2, save_path=None):
    glitches_list = EventTable.fetch("gravityspy", "glitches_v2d0",
                                     selection=["0.9<=ml_confidence<=1.0",
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


if __name__ == "__main__":
    # t1, t2 = to_gps("2019-04-01"), to_gps("2020-03-27")
    # glitches = get_glitches(t1, t2, "./glitches.csv")
    glitches = pd.read_csv("./glitches.csv")

    # write sub
    sub_name = "autosl.sub"
    job_sub = sub_file.SubFile(sub_name)
    job_sub.add_executable(os.path.join(ANALYSIS_PATH, "job.py"))
    job_sub.add_arguments("--ifo $(IFO) --channel $(CHN) --ml_label $(MLB) --peak_time $(PKT) "
                          "--peak_freq $(PKF) --opath {}".format(ANALYSIS_PATH))
    job_sub.add_accounting_group_info(ACCOUNTING_GROUP, os.path.expandvars("$USER"))
    job_sub.add_specs(3, 1000, disk=20000)
    job_sub.add("periodic_remove = (time() - EnteredCurrentStatus) > 3600")
    job_logs = os.path.join(ANALYSIS_PATH, "logs")
    os.system("mkdir -p {}".format(job_logs))
    job_sub.add_logs(job_logs, job_logs, ["$(PKT)"])
    job_sub.save()

    # write dag
    dag_name = "autosl.dag"
    dag = dag_file.DagFile(dag_name)
    for i, g in glitches.iterrows():
        dag.add_job(i + 1, sub_name, args={"IFO": g.ifo, "CHN": g.channel,
                                           "MLB": g.ml_label, "PKT": g.peak_time,
                                           "PKF": g.peak_frequency})
    dag.save()

    # delete previous dag output files and submit dag file
    os.system("rm -rf {}.*".format(dag_name))
    os.system("condor_submit_dag -maxjobs 100 {}".format(dag_name))
