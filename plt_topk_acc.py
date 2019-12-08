# matplotlib
import matplotlib.pyplot as plt
import json
from os.path import join
import sys
import pandas as pd
# plt.switch_backend('Agg')

exp_folder = sys.argv[1]
print("Experiment result folder:", exp_folder)

# load json file first
file = join(exp_folder, "search_acc_per_class.json")
with open(file, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df = df.set_index("catagory")
df = df[["mean", "std"]]

ax = df.plot(kind="bar")
ax.xaxis.set_tick_params(rotation=45)

fig = ax.get_figure()

png = join(exp_folder, "search_acc.png")
fig.savefig(png, bbox_inches='tight')
pdf = join(exp_folder, "search_acc.pdf")
fig.savefig(pdf, bbox_inches='tight')