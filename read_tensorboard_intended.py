from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import matplotlib.pyplot as plt

def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}




scalars = [
    "real/big",
    "freq/big",
]

mydict = {}
ppwDict = {}
myfpsDict = {}
tempDict = {}
for subdir, dirs, files in os.walk("selected/intended"):
    a = subdir.split("/")

    

    if len(a) > 2:
        mykey = a[2].split("_")[2] + a[2].split("_")[4]
        if mykey not in mydict:
            mydict[mykey] = []
            ppwDict[mykey] = []
            myfpsDict[mykey] = []
            tempDict[mykey] = []

for subdir, dirs, files in os.walk("selected/intended"):
    # print(subdir)
    for file in files:

        a = subdir.split("/")
        if len(a) > 2:
            mykey = a[2].split("_")[2] + a[2].split("_")[4]

        filepath = subdir + os.sep + file
        if "events" in filepath:
            x = parse_tensorboard(filepath, scalars)
            # print(subdir, x["perf/ppw"][-50:]["value"].mean())
            # print(subdir, x["perf/fps"][-10:]["value"].mean())

            mydict[mykey].append(x["real/big"])
            mydict[mykey].append(x["freq/big"])
            
            # mydict[mykey] = 

df = pd.DataFrame()



import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib import font_manager
font_path = '/usr/share/fonts/truetype/msttcorefonts/arial.ttf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()
matplotlib.rcParams.update({'font.size': 22})



fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))

deft = mydict["zTT21"][0]
deft2 = []
deft3 = []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000000)

limit = mydict["zTT21"][1]
limit2 = []
limit3 = []
firstTime = limit["wall_time"][0]
for i in range(len(limit["step"])):
    if i %2 == 1:
        limit2.append(limit["wall_time"][i] - firstTime)
        limit3.append(sum(limit["value"][i-1:i+1]) / 2/1000000)

axes[0].plot(limit2[:16], limit3[:16], label = "Action")
axes[0].plot(deft2[:16], deft3[:16], label = "Real")
axes[0].set_title("zTT", fontsize=22)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Frequency (Ghz)")
axes[0].set_ylim(1.02,2.9)
axes[0].set_xlim(-10,310)

deft = mydict["gearDVFS1"][0]
deft2 = []
deft3 = []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000000)

limit = mydict["gearDVFS1"][1]
limit2 = []
limit3 = []
firstTime = limit["wall_time"][0]
for i in range(len(limit["step"])):
    if i %2 == 1:
        limit2.append(limit["wall_time"][i] - firstTime)
        limit3.append(sum(limit["value"][i-1:i+1]) / 2 /1000000)

axes[1].plot(limit2[:31], limit3[:31], label = "Action")
axes[1].plot(deft2[:31], deft3[:31], label = "Real")
axes[1].set_title("gearDVFS", fontsize=22)
axes[1].set_yticks([])

axes[1].set_xlabel("Time (s)")
# axes[1].set_ylabel("Frequency (Ghz)")
axes[1].set_xlim(-10,310)
axes[1].set_ylim(1.02,2.9)
axes[1].legend(loc = "upper right", borderpad = 0.3, labelspacing=0.2, borderaxespad=0.3, handletextpad=0.4, handlelength=1.5)
plt.tight_layout(pad = 0.5)

plt.show()



gearReal = mydict["gearDVFS1"][0]
gearIntended = mydict["gearDVFS1"][1]

zTTReal = mydict["zTT21"][0]
zTTIntended = mydict["zTT21"][1]



plt.plot(zTTReal[:30], label = "zTT real freq")
plt.plot(zTTIntended[:30], label = "zTT action freq")

plt.plot(gearReal[:30], label = "gearDVFS real freq")
plt.plot(gearIntended[:30], label = "gearDVFS action freq")
plt.legend()
plt.tight_layout()
plt.show()