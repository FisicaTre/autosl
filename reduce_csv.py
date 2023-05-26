import pandas as pd

MAX_SELECT = 10

glitches = pd.read_csv("glitches.csv")
g_types = glitches.ml_label.unique()
new_df = pd.DataFrame([])
for g_type in g_types:
    g_df = glitches[glitches.ml_label == g_type]
    n_g_type = len(g_df)
    if n_g_type < MAX_SELECT:
        new_df = pd.concat([new_df, g_df])
    else:
        new_df = pd.concat([new_df, g_df.sample(MAX_SELECT)])
new_df.to_csv("glitches_short.csv")
