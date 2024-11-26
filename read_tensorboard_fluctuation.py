from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# import PyQt5
# import matplotlib
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
    "cstate/big",
    "cstate/mid",
    "cstate/little",
    "cstate/gpu",
    "temp/battery",
    "temp/big",
    "temp/mid",
    "temp/little",
    "temp/gpu",
    "freq/big",
    "freq/mid",
    "freq/little",
    "freq/gpu",
    "util/big",

]

mydict = {}
ppwDict = {}
myfpsDict = {}
tempDict = {}
batteryDict = {}
powerDict = {}
cpuDict = {}

bigD = {}
midD = {}
litD = {}
gpuD = {}
utilD = {}
lossD = {}
rewardD = {}

for subdir, dirs, files in os.walk("selected/fluctuation"):
    a = subdir.split("/")


    if len(a) > 2:
        if len(a) > 2:
            if "def" in a[2].split("_")[2]:
                mykey = a[2].split("_")[2] + a[2].split("_")[-1]
            else:
                mykey = a[2].split("_")[2] + a[2].split("_")[-1]

        if mykey not in mydict:
            mydict[mykey] = []
            ppwDict[mykey] = []
            myfpsDict[mykey] = []
            tempDict[mykey] = []    
            batteryDict[mykey] = []    
            powerDict[mykey] = []    
            cpuDict[mykey] = []   
                        
            bigD[mykey] = []    
            midD[mykey] = []    
            litD[mykey] = []    
            gpuD[mykey] = []    
            utilD[mykey] = []    
            lossD[mykey] = []    
            rewardD[mykey] = []    
 


for subdir, dirs, files in os.walk("selected/fluctuation"):
    # print(subdir)
    for file in files:

        a = subdir.split("/")
        if len(a) > 2:
            if "def" in a[2].split("_")[2]:
                mykey = a[2].split("_")[2] + a[2].split("_")[-1]
            else:
                mykey = a[2].split("_")[2] + a[2].split("_")[-1]

        filepath = subdir + os.sep + file
        if "events" in filepath:
            x = parse_tensorboard(filepath, scalars)
            print(subdir, x["perf/ppw"][-50:]["value"].mean())
            # print(subdir, x["perf/fps"][-10:]["value"].mean())

            mydict[mykey].append(x["perf/ppw"])
            
            myfpsDict[mykey].append(x["perf/fps"])

            batteryDict[mykey].append(x["temp/battery"])
            
            powerDict[mykey].append(x["perf/power"])



            cstate = x["cstate/big"]
            cstate["value"] = cstate["value"] + x["cstate/mid"]["value"]
            cstate["value"] = cstate["value"] + x["cstate/little"]["value"]
            cstate["value"] = cstate["value"] + x["cstate/gpu"]["value"]

            cputemp = x["temp/big"]
            cputemp["value"] = cputemp["value"] + x["temp/mid"]["value"]
            cputemp["value"] = cputemp["value"] + x["temp/little"]["value"] + x["temp/gpu"]["value"]

            cpuDict[mykey].append(cputemp)

            tempDict[mykey].append(cstate)

            bigD[mykey].append(x["freq/big"])
            midD[mykey].append(x["freq/mid"])
            litD[mykey].append(x["freq/little"])
            gpuD[mykey].append(x["freq/gpu"])
            utilD[mykey].append(x["util/big"])

            if "DVFStrain" in mykey:
                x = parse_tensorboard(filepath, ["losses/qf_loss"])
            elif "zTT" in mykey:
                x = parse_tensorboard(filepath, ["losses/td_loss"])
            elif "gear" in mykey:
                x = parse_tensorboard(filepath, ["losses/loss"])
            else:
                x = 1

            lossD[mykey].append(x)

            if "DVFStrain" in mykey:
                x = parse_tensorboard(filepath, ["perf/reward"])
            elif "zTT" in mykey:
                x = parse_tensorboard(filepath, ["perf/reward"])
            elif "gear" in mykey:
                x = parse_tensorboard(filepath, ["perf/reward"])
            else:
                x = 1

            rewardD[mykey].append(x)




mydict.keys()



1 - batteryDict['limitDefaultgpu400000'][0].value.mean()/batteryDict['defaulttemp20'][0].value.mean()
1 - powerDict['limitDefaultgpu400000'][0].value.mean()/powerDict['defaulttemp20'][0].value.mean()


1 - cpuDict['limitDefaultgpu400000'][0].value.mean()/cpuDict['defaulttemp20'][0].value.mean()
1 - tempDict['limitDefaultgpu400000'][0].value.mean()/tempDict['defaulttemp20'][0].value.mean()



############ figure 1

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





cmap = plt.get_cmap("Set1")


# fig, ax1 = plt.subplots(figsize=(10, 3.6))
fig, ax1 = plt.subplots(figsize=(6, 3.6))

# First dataset
deft = myfpsDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i % 2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)

# Plot first dataset on primary y-axis
ax1.plot(deft2[:46], deft3[:46], color=cmap(0), label="fps")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frame rates (fps)", color=cmap(0))
ax1.tick_params(axis="y", labelcolor=cmap(0))
ax1.set_ylim(30, 48)
ax1.set_yticks([30, 36, 42, 48])


# Second dataset
deft = tempDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i % 2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)

# Create a second y-axis
ax2 = ax1.twinx()
ax2.plot(deft2[:46], deft3[:46], color=cmap(1), label="c-states")
ax2.set_ylabel("Total cooling states", color=cmap(1))
ax2.tick_params(axis="y", labelcolor=cmap(1))

ax2.set_xticks([0,300,600,900,1200])
ax2.set_ylim(0, 30)
ax2.set_yticks([0, 10, 20, 30])
ax1.grid(axis='both',linestyle='--')  # Add grid to the second subplot
ax1.set_axisbelow(True)

# Add a legend
fig.tight_layout(pad=0.5)  # Adjust layout to prevent overlap
plt.show()


fig, ax1 = plt.subplots(figsize=(6, 3.6))

# First dataset
deft = myfpsDict['gearDVFStargetUtil0.0'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i % 4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)

# Plot first dataset on primary y-axis
ax1.plot(deft2[:61], deft3[:61], color=cmap(0), label="fps")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Frame rates (fps)", color=cmap(0))
ax1.tick_params(axis="y", labelcolor=cmap(0))
ax1.set_ylim(30, 48)
ax1.set_yticks([30, 36, 42, 48])


# Second dataset
deft = tempDict['gearDVFStargetUtil0.0'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i % 4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)

# Create a second y-axis
ax2 = ax1.twinx()
ax2.plot(deft2[:61], deft3[:61], color=cmap(1), label="c-states")
ax2.set_ylabel("Total cooling states", color=cmap(1))
ax2.tick_params(axis="y", labelcolor=cmap(1))

ax2.set_xticks([0,300,600,900,1200])
# ax2.set_ylim(5, 35)
# ax2.set_yticks([5, 15, 25, 35])
ax2.set_ylim(0, 30)
ax2.set_yticks([0, 10, 20, 30])
ax1.grid(axis='both',linestyle='--')  # Add grid to the second subplot
ax1.set_axisbelow(True)

# Add a legend
fig.tight_layout(pad=0.5)  # Adjust layout to prevent overlap
plt.show()




############ figure 2


cmap = plt.get_cmap("tab20c")





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


fig, axes = plt.subplots(2, 2, figsize=(10, 3.7))


deft = myfpsDict['defaulttemp20'][0]
deft2 = []
deft3 = []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)

limit = myfpsDict['limitDefaultgpu400000'][0]
limit2 = []
limit3 = []
firstTime = limit["wall_time"][0]
for i in range(len(limit["step"])):
    if i %2 == 1:
        limit2.append(limit["wall_time"][i] - firstTime)
        limit3.append(sum(limit["value"][i-1:i+1]) / 2)

axes[0][0].plot(deft2, deft3, color = cmap(0))
axes[0][0].plot(limit2, limit3, color = cmap(16))
axes[0][0].set_xlabel("Frame rates (fps)")
axes[0][0].set_xticks([])

deft = powerDict['defaulttemp20'][0]
deft2 = []
deft3 = []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000)

limit = powerDict['limitDefaultgpu400000'][0]
limit2 = []
limit3 = []
firstTime = limit["wall_time"][0]
for i in range(len(limit["step"])):
    if i %2 == 1:
        limit2.append(limit["wall_time"][i] - firstTime)
        limit3.append(sum(limit["value"][i-1:i+1]) / 2 / 1000)

axes[0][1].plot(deft2, deft3, color = cmap(0))
axes[0][1].plot(limit2, limit3, color = cmap(16))
axes[0][1].set_xlabel("Power consumption (W)")
axes[0][1].set_xticks([])


deft = tempDict['defaulttemp20'][0]
deft2 = []
deft3 = []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)

limit = tempDict['limitDefaultgpu400000'][0]
limit2 = []
limit3 = []
firstTime = limit["wall_time"][0]
for i in range(len(limit["step"])):
    if i %2 == 1:
        limit2.append(limit["wall_time"][i] - firstTime)
        limit3.append(sum(limit["value"][i-1:i+1]) / 2 )

axes[1][0].plot(deft2, deft3, color = cmap(0))
axes[1][0].plot(limit2, limit3, color = cmap(16))
axes[1][0].set_xlabel("Total cooling states")
axes[1][0].set_xticks([])


deft = cpuDict['defaulttemp20'][0]
deft2 = []
deft3 = []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2/4)

limit = cpuDict['limitDefaultgpu400000'][0]
limit2 = []
limit3 = []
firstTime = limit["wall_time"][0]
for i in range(len(limit["step"])):
    if i %2 == 1:
        limit2.append(limit["wall_time"][i] - firstTime)
        limit3.append(sum(limit["value"][i-1:i+1]) / 2 /4)



axes[1][1].plot(deft2, deft3, color = cmap(0))
axes[1][1].plot(limit2, limit3, color = cmap(16))
axes[1][1].set_xlabel("Avg. processor temperature (°C)", fontsize=22)
axes[1][1].set_xticks([])


axes[0][0].grid(axis='y',linestyle='--')  
axes[0][0].set_axisbelow(True)

axes[0][1].grid(axis='y',linestyle='--')  
axes[0][1].set_axisbelow(True)

axes[1][0].grid(axis='y',linestyle='--')  
axes[1][0].set_axisbelow(True)

axes[1][1].grid(axis='y',linestyle='--')  
axes[1][1].set_axisbelow(True)

axes[1][0].set_yticks([0,12,24])
axes[1][1].set_yticks([70,75,80])


fig.legend(["Default governor", "Proactive throttling"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.047),frameon=False, borderpad=0.1, fontsize=22)
plt.tight_layout(pad = 0.05, rect = (0,0,1,0.92))

plt.subplots_adjust(wspace=0.15)

plt.show()




############ figure 3


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


fig, axes = plt.subplots(2, 2, figsize=(10, 4))


deft = myfpsDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3)

deft = myfpsDict['zTT2target65'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
axes[0][0].plot(deft2, deft3)
axes[0][0].set_xlabel("Frame rates (fps)")
axes[0][0].set_xticks([])

deft = myfpsDict['gearDVFStargetUtil0.0'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
axes[0][0].plot(deft2, deft3)
axes[0][0].set_xlabel("Frame rates (fps)")
axes[0][0].set_xticks([])

deft = myfpsDict['DVFStrain11temp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
axes[0][0].plot(deft2, deft3)
axes[0][0].set_xlabel("Frame rates (fps)")
axes[0][0].set_xticks([])





deft = powerDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000)
axes[0][1].plot(deft2, deft3)

deft = powerDict['zTT2target65'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 1000)
axes[0][1].plot(deft2, deft3)
axes[0][1].set_xlabel("Power consumption (W)")
axes[0][1].set_xticks([])

deft = powerDict['gearDVFStargetUtil0.0'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 1000)
axes[0][1].plot(deft2, deft3)
axes[0][1].set_xlabel("Power consumption (W)")
axes[0][1].set_xticks([])

deft = powerDict['DVFStrain11temp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 1000)
axes[0][1].plot(deft2, deft3)
axes[0][1].set_xlabel("Power consumption (W)")
axes[0][1].set_xticks([])




deft = tempDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][0].plot(deft2, deft3)

deft = tempDict['zTT2target65'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
axes[1][0].plot(deft2, deft3)
axes[1][0].set_xlabel("Total cooling states")
axes[1][0].set_xticks([])

deft = tempDict['gearDVFStargetUtil0.0'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
axes[1][0].plot(deft2, deft3)
axes[1][0].set_xlabel("Total cooling states")
axes[1][0].set_xticks([])

deft = tempDict['DVFStrain11temp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
axes[1][0].plot(deft2, deft3)
axes[1][0].set_xlabel("Total cooling states")
axes[1][0].set_xticks([])




deft = cpuDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 4)
axes[1][1].plot(deft2, deft3)

deft = cpuDict['zTT2target65'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 4)
axes[1][1].plot(deft2, deft3)
axes[1][1].set_xlabel("CPU temperature (°C)")
axes[1][1].set_xticks([])


deft = cpuDict['gearDVFStargetUtil0.0'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 4)
axes[1][1].plot(deft2, deft3)
axes[1][1].set_xlabel("CPU temperature (°C)")
axes[1][1].set_xticks([])


deft = cpuDict['DVFStrain11temp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 4)
axes[1][1].plot(deft2, deft3)
axes[1][1].set_xlabel("Processor temperature (°C)")
axes[1][1].set_xticks([])

# fig.legend(["Default", "Limit"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, fontsize=22)
fig.legend(["Default", "zTT", "gear", "ear"], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, fontsize=22)
plt.tight_layout(pad = 0.5, rect = (0,0,1,0.95))

plt.show()










############ figure 4

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


fig, axes = plt.subplots(2, 3, figsize=(21, 5))


deft = myfpsDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %1 == 0:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-0:i+1]) / 1)
axes[0][0].plot(deft2, deft3)

deft = myfpsDict['zTT2target65'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3)
axes[0][0].set_xlabel("Frame rates (fps)")
axes[0][0].set_xticks([])

deft = myfpsDict['gearDVFStargetUtil0.0'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3)
axes[0][0].set_xlabel("Frame rates (fps)")
axes[0][0].set_xticks([])

deft = myfpsDict['DVFStrain11temp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3)
axes[0][0].set_xlabel("Frame rates (fps)")
axes[0][0].set_xticks([])





deft = powerDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %1 == 0:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-0:i+1]) / 1 / 1000)
axes[0][1].plot(deft2, deft3)

deft = powerDict['zTT2target65'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000)
axes[0][1].plot(deft2, deft3)
axes[0][1].set_xlabel("Power consumption (W)")
axes[0][1].set_xticks([])

deft = powerDict['gearDVFStargetUtil0.0'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000)
axes[0][1].plot(deft2, deft3)
axes[0][1].set_xlabel("Power consumption (W)")
axes[0][1].set_xticks([])

deft = powerDict['DVFStrain11temp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000)
axes[0][1].plot(deft2, deft3)
axes[0][1].set_xlabel("Power consumption (W)")
axes[0][1].set_xticks([])




deft = tempDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %1 == 0:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-0:i+1]) / 1)
axes[0][2].plot(deft2, deft3)

deft = tempDict['zTT2target65'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][2].plot(deft2, deft3)
axes[0][2].set_xlabel("Total cooling states")
axes[0][2].set_xticks([])

deft = tempDict['gearDVFStargetUtil0.0'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][2].plot(deft2, deft3)
axes[0][2].set_xlabel("Total cooling states")
axes[0][2].set_xticks([])

deft = tempDict['DVFStrain11temp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][2].plot(deft2, deft3)
axes[0][2].set_xlabel("Total cooling states")
axes[0][2].set_xticks([])




deft = cpuDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %1 == 0:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-0:i+1]) / 1 / 4)
axes[1][0].plot(deft2, deft3)

deft = cpuDict['zTT2target65'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 4)
axes[1][0].plot(deft2, deft3)
axes[1][0].set_xlabel("CPU temperature (°C)")
axes[1][0].set_xticks([])


deft = cpuDict['gearDVFStargetUtil0.0'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 4)
axes[1][0].plot(deft2, deft3)
axes[1][0].set_xlabel("CPU temperature (°C)")
axes[1][0].set_xticks([])


deft = cpuDict['DVFStrain11temp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 4)
axes[1][0].plot(deft2, deft3)
axes[1][0].set_xlabel("Processor temperature (°C)")
axes[1][0].set_xticks([])




key= list(lossD['zTT2target65'][0].keys())[0]
deft = lossD['zTT2target65'][0][key]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 4)
deft3 = np.array(deft3)
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
axes[1][1].plot(deft2, deft3, color = "C1")
axes[1][1].set_xlabel("CPU temperature (°C)")
axes[1][1].set_xticks([])

key= list(lossD['gearDVFStargetUtil0.0'][0].keys())[0]
deft = lossD['gearDVFStargetUtil0.0'][0][key]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 4)
deft3 = np.array(deft3)
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
axes[1][1].plot(deft2, deft3, color = "C2")
axes[1][1].set_xlabel("CPU temperature (°C)")
axes[1][1].set_xticks([])


key= list(lossD['DVFStrain11temp20'][0].keys())[0]
deft = lossD['DVFStrain11temp20'][0][key]

deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 4)

deft3 = np.array(deft3)
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
axes[1][1].plot(deft2, deft3, color = "C3")
axes[1][1].set_xlabel("Normalized loss")
axes[1][1].set_xticks([])





key= list(rewardD['zTT2target65'][0].keys())[0]
deft = rewardD['zTT2target65'][0][key]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 4)
deft3 = np.array(deft3)
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
axes[1][2].plot(deft2, deft3, color = "C1")
axes[1][2].set_xlabel("CPU temperature (°C)")
axes[1][2].set_xticks([])

key= list(rewardD['gearDVFStargetUtil0.0'][0].keys())[0]
deft = rewardD['gearDVFStargetUtil0.0'][0][key]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 4)
deft3 = np.array(deft3)
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
axes[1][2].plot(deft2, deft3, color = "C2")
axes[1][2].set_xlabel("CPU temperature (°C)")
axes[1][2].set_xticks([])


key= list(rewardD['DVFStrain11temp20'][0].keys())[0]
deft = rewardD['DVFStrain11temp20'][0][key]

deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 4)

deft3 = np.array(deft3)
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
axes[1][2].plot(deft2, deft3, color = "C3")
axes[1][2].set_xlabel("Normalized reward")
axes[1][2].set_xticks([])



# fig.legend(["Default", "Limit"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, fontsize=22)
fig.legend(["Default", "zTT", "gear", "ear"], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, fontsize=22)
plt.tight_layout(pad = 0.5, rect = (0,0,1,0.95))

plt.show()









############ figure 5


fig, axes = plt.subplots(1, 1, figsize=(6, 3.2))

key= list(lossD['zTT2target65'][0].keys())[0]
deft = lossD['zTT2target65'][0][key]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
deft3 = np.array(deft3)
# deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
# deft3 = (deft3) / (np.linalg.norm(deft3))
# deft3 = (deft3) / deft3.max()
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())

axes.plot(deft2, deft3)

key= list(lossD['gearDVFStargetUtil0.0'][0].keys())[0]
deft = lossD['gearDVFStargetUtil0.0'][0][key]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
deft3 = np.array(deft3)
# deft3 = deft3 / deft3.mean()
# deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
# deft3 = (deft3) / deft3.max()
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
    
axes.plot(deft2, deft3)

key= list(lossD['DVFStrain11temp20'][0].keys())[0]
deft = lossD['DVFStrain11temp20'][0][key]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
deft3 = np.array(deft3)
# deft3 = deft3 / deft3.mean()
# deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
# deft3 = (deft3) / deft3.max()
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
    
axes.plot(deft2, deft3)
axes.set_xlabel("Time (s)")
axes.set_ylabel("Normalized loss")
# axes[0].set_xticks([])


fig.legend(["zTT", "gear", "ear"], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1)
plt.tight_layout(pad = 0.5, rect = (0,0,1,0.925))

plt.show()



############ figure 6


fig, axes = plt.subplots(1, 1, figsize=(6, 3.2))

key= list(rewardD['zTT2target65'][0].keys())[0]
deft = rewardD['zTT2target65'][0][key]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
deft3 = np.array(deft3)
# deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
# deft3 = (deft3) / (np.linalg.norm(deft3))
# deft3 = (deft3) / deft3.max()
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())

axes.plot(deft2, deft3)

key= list(rewardD['gearDVFStargetUtil0.0'][0].keys())[0]
deft = rewardD['gearDVFStargetUtil0.0'][0][key]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
deft3 = np.array(deft3)
# deft3 = deft3 / deft3.mean()
# deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
# deft3 = (deft3) / deft3.max()
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
    
axes.plot(deft2, deft3)

key= list(rewardD['DVFStrain11temp20'][0].keys())[0]
deft = rewardD['DVFStrain11temp20'][0][key]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
deft3 = np.array(deft3)
# deft3 = deft3 / deft3.mean()
# deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
# deft3 = (deft3) / deft3.max()
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
    
axes.plot(deft2, deft3)
axes.set_xlabel("Time (s)")
axes.set_ylabel("Normalized reward")
# axes[0].set_xticks([])


fig.legend(["zTT", "gear", "ear"], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1)
plt.tight_layout(pad = 0.5, rect = (0,0,1,0.925))

plt.show()







ours =  tempDict['DVFStrain11temp20'][0]
deft = tempDict['defaulttemp20'][0]
limit = tempDict["limitDefaultgpu400000"][0]

ours =  mydict['DVFStrain11temp20'][0]
deft = mydict['defaulttemp20'][0]



ours =  myfpsDict['DVFStrain11temp20'][0]
deft = myfpsDict['defaulttemp20'][0]

ours2 = []
ours3 = []

firstTime = ours["wall_time"][0]
for i in range(len(ours["step"])):
    if i %4 == 3:
        ours2.append(ours["wall_time"][i] - firstTime)
        ours3.append(sum(ours["value"][i-3:i+1]) / 4)


deft2 = []
deft3 = []

firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)

plt.figure(figsize=(8,4.5))
plt.plot(ours2, ours3, label = "ours")
plt.plot(deft2, deft3, label = "default")
plt.legend()
plt.show()