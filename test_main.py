import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import bags_array
import approximate


# ---------- данные ----------
folder = r"C:\Users\misha\Desktop\data\data\clear_water\F33\F33_20000"

bags_array = bags_array.bags_array(folder)
df = bags_array[12].copy()

approximate.appr(df)