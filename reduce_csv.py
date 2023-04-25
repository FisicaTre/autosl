import pandas as pd

glitches = pd.read_csv("glitches.csv")
g_types = glitches.ml_label.unique()
print(g_types)
