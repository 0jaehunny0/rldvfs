from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# import PyQt5
# matplotlib.use('Qt5Agg')

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
    "perf/fps",
    "perf/ppw",
    "perf/power",
    "temp/gpu",
    "temp/big",
    "temp/mid",
    "temp/little",
]


mydict = {}
ppwDict = {}
myfpsDict = {}
tempDict = {}
powerDict = {}
for subdir, dirs, files in os.walk("selected/proctemp"):
    a = subdir.split("/")


    if len(a) > 2:
        if len(a) > 2:
            if "def" in a[2].split("_")[2]:
                mykey = a[2].split("_")[2][:3] + a[2].split("_")[4]
            else:
                mykey = a[2].split("_")[2][:3] + a[2].split("_")[6]

        if mykey not in mydict:
            mydict[mykey] = []
            ppwDict[mykey] = []
            myfpsDict[mykey] = []
            tempDict[mykey] = []    
            powerDict[mykey] = []    


for subdir, dirs, files in os.walk("selected/proctemp"):
    # print(subdir)
    for file in files:

        a = subdir.split("/")
        if len(a) > 2:
            if "def" in a[2].split("_")[2]:
                mykey = a[2].split("_")[2][:3] + a[2].split("_")[4]
            else:
                mykey = a[2].split("_")[2][:3] + a[2].split("_")[6]

        filepath = subdir + os.sep + file
        if "events" in filepath:
            x = parse_tensorboard(filepath, scalars)
            print(subdir, x["perf/ppw"][-50:].mean())
            # print(subdir, x["perf/fps"][-10:].mean())

            cputemp = x["temp/big"]
            cputemp["value"] = cputemp["value"] + x["temp/mid"]["value"]
            cputemp["value"] = cputemp["value"] + x["temp/little"]["value"] + x["temp/gpu"]["value"]


            mydict[mykey].append(x["perf/ppw"][:])
            powerDict[mykey].append(x["perf/power"][:])
            myfpsDict[mykey].append(x["perf/fps"][:])
            # tempDict[mykey].append(x["temp/gpu"][:])
            tempDict[mykey].append(cputemp)

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

#################3


cmap = plt.get_cmap("tab20c")

tempDict['freexp1'][0]["value"][12:]/4
powerDict['freexp1'][0]["value"][12:]/1000
########################
import matplotlib.pyplot as plt
import numpy as np

(tempDict['freexp1'][0]["value"][12:]/4).iloc[12]

(tempDict['freexp1'][0]["value"][12:]/4).iloc[142]

cmap = plt.get_cmap("tab20c")
fig, ax= plt.subplots(figsize=(6, 3.4))
plt.scatter(tempDict['freexp1'][0]["value"][12:]/4, powerDict['freexp1'][0]["value"][12:]/1000, color = "C4")
plt.xlabel("Avg. processor temperature (°C)")
# plt.ylabel("Power consumption (W)")

ax.text(-0.1, 0.37, 'Power consumption (W)', rotation=90, 
            verticalalignment='center', horizontalalignment='right', 
            transform=ax.transAxes)


ax.set_ylim(0.9, 1.25)
ax.set_yticks([0.9,1.0,1.1, 1.2])
plt.grid(axis='both',linestyle='--')  # Add grid to the second subplot
ax.set_axisbelow(True)
plt.tight_layout(pad=0.3)  # Adjust layout to prevent overlap
plt.show()



fig, ax1 = plt.subplots(figsize=(10, 3.6))

# First dataset
deft = myfpsDict['freexp1'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i % 2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)

# Plot first dataset on primary y-axis
ax1.plot(deft2, deft3, color="r", label="fps")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frame rates (fps)", color="r")
ax1.tick_params(axis="y", labelcolor="r")

# Second dataset
deft = powerDict['freexp1'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i % 2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)

# Create a second y-axis
ax2 = ax1.twinx()
ax2.plot(deft2, deft3, color="b", label="c-states")
ax2.set_ylabel("Power consumption", color="b")
ax2.tick_params(axis="y", labelcolor="b")

# Add a legend
fig.tight_layout(pad=0.5)  # Adjust layout to prevent overlap
plt.show()




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


fig, axes = plt.subplots(3, 1, figsize=(6, 3))

plt.subplots_adjust(hspace=0.0)

deft = myfpsDict['freexp1'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
axes[0].plot(deft2[:23], deft3[:23])
axes[0].yaxis.tick_right() 
axes[0].set_xticks([])
axes[0].set_ylim(40,45)
axes[0].text(-0.1, 0.5, 'Frame rates     \n(fps)          ', rotation=0, 
            verticalalignment='center', horizontalalignment='right', 
            transform=axes[0].transAxes)
# axes[0].grid(axis='y',linestyle='--')  # Add grid to the second subplot

deft = powerDict['freexp1'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 /1000)
axes[1].plot(deft2[:23], deft3[:23])
axes[1].yaxis.tick_right() 
axes[1].set_xticks([])
axes[1].text(-0.1, 0.5, 'Power         \nconsumption (W)', rotation=0, 
            verticalalignment='center', horizontalalignment='right', 
            transform=axes[1].transAxes)
# axes[1].grid(axis='y',linestyle='--')  # Add grid to the second subplot

deft = tempDict['freexp1'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 4)
axes[2].plot(deft2[:23], deft3[:23])
axes[2].yaxis.tick_right() 
axes[2].set_xticks([])
axes[2].text(-0.1, 0.5, 'Avg. processor  \ntemperature (°C)', rotation=0, 
            verticalalignment='center', horizontalalignment='right', 
            transform=axes[2].transAxes)
# axes[2].grid(axis='y',linestyle='--')  # Add grid to the second subplot


plt.tight_layout(pad = 0.5, rect = (0,0,1,1))
plt.show()

