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






# fig, ax1 = plt.subplots(figsize=(10, 3.6))

cmap = plt.get_cmap("tab20c")



fig, ax1 = plt.subplots(figsize=(6, 3))


deft = mydict['defaulttemp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 * 1000)
ax1.plot(deft2, deft3, color = cmap(0), alpha = 0.95, label = "Deft", linestyle = "--")

deft = mydict['zTT2target65'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 * 1000)
ax1.plot(deft2, deft3, color = cmap(4), alpha = 0.95, label = "zTT", linestyle = "-.")

deft = mydict['gearDVFStargetUtil0.0'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 * 1000)
ax1.plot(deft2, deft3, color = cmap(8), alpha = 0.95, label = "Gear", linestyle = (0, (1, 0.5)))

deft = mydict['DVFStrain11temp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 * 1000)
ax1.plot(deft2, deft3, color = cmap(12), alpha = 0.95, label = "Ear")
ax1.set_ylabel("PPW")
ax1.set_xlabel("Time (s)")
ax1.grid(linestyle = '--', axis='both')
plt.tight_layout(pad = 0.1)

ax1.set_ylim(ax1.get_ylim()[0] , 37)
plt.legend(ncol = 4, borderpad = 0.2, labelspacing=0.2, borderaxespad=0.2, handletextpad=0.4, handlelength=1.105, columnspacing = 0.8, loc = "upper center", fancybox = False)


plt.show()

############ 


fig, ax1 = plt.subplots(figsize=(6, 3))


deft = myfpsDict['defaulttemp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
ax1.plot(deft2, deft3, color = cmap(0), alpha = 0.95, linestyle = "--")

deft = myfpsDict['zTT2target65'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
ax1.plot(deft2, deft3, color = cmap(4), alpha = 0.95, linestyle = "-.")

deft = myfpsDict['gearDVFStargetUtil0.0'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
ax1.plot(deft2, deft3, color = cmap(8), alpha = 0.95, linestyle = (0, (1, 0.5)))

deft = myfpsDict['DVFStrain11temp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
ax1.plot(deft2, deft3, color = cmap(12), alpha = 0.95)
ax1.set_ylabel("QoE")
ax1.set_xlabel("Time (s)")
ax1.grid(linestyle = '--', axis='both')
plt.tight_layout(pad = 0.1)
plt.show()



############ 


fig, ax1 = plt.subplots(figsize=(6, 3))


deft = cpuDict['defaulttemp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 /4)
ax1.plot(deft2, deft3, color = cmap(0), alpha = 0.95, linestyle = "--")

deft = cpuDict['zTT2target65'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 /4)
ax1.plot(deft2, deft3, color = cmap(4), alpha = 0.95, linestyle = "-.")

deft = cpuDict['gearDVFStargetUtil0.0'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 /4)
ax1.plot(deft2, deft3, color = cmap(8), alpha = 0.95, linestyle = (0, (1, 0.5)))

deft = cpuDict['DVFStrain11temp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 /4)
ax1.plot(deft2, deft3, color = cmap(12), alpha = 0.95)
ax1.set_ylabel("Avg. proc. temp.")
ax1.set_xlabel("Time (s)")
ax1.grid(linestyle = '--', axis='both')
plt.tight_layout(pad = 0.1)
plt.show()


############ 

fig, ax1 = plt.subplots(figsize=(6, 3))


deft = tempDict['defaulttemp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
ax1.plot(deft2, deft3, color = cmap(0), alpha = 0.95, linestyle = "--")

deft = tempDict['zTT2target65'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
ax1.plot(deft2, deft3, color = cmap(4), alpha = 0.95, linestyle = "-.")

deft = tempDict['gearDVFStargetUtil0.0'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
ax1.plot(deft2, deft3, color = cmap(8), alpha = 0.95, linestyle = (0, (1, 0.5)))

deft = tempDict['DVFStrain11temp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
ax1.plot(deft2, deft3, color = cmap(12), alpha = 0.95)
ax1.set_ylabel("Cool. states.")
ax1.set_xlabel("Time (s)")
ax1.grid(linestyle = '--', axis='both')
plt.tight_layout(pad = 0.1)
plt.show()

############ 



fig, ax1 = plt.subplots(figsize=(6, 3))

key = list(lossD['zTT2target65'][0].keys())[0]
deft = lossD['zTT2target65'][0][key]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
deft3 = np.array(deft3)
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
ax1.plot(deft2, deft3, color = cmap(4), alpha = 0.95, label = "zTT", linestyle = "-.")

key = list(lossD['gearDVFStargetUtil0.0'][0].keys())[0]
deft = lossD['gearDVFStargetUtil0.0'][0][key]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
deft3 = np.array(deft3)
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
ax1.plot(deft2, deft3, color = cmap(8), alpha = 0.95, label = "Gear", linestyle = (0, (1, 0.5)))

key = list(lossD['DVFStrain11temp20'][0].keys())[0]
deft = lossD['DVFStrain11temp20'][0][key]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
deft3 = np.array(deft3)
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
ax1.plot(deft2, deft3, color = cmap(12), alpha = 0.95, label = "Ear")
ax1.set_ylabel("Loss")
ax1.set_xlabel("Time (s)")
ax1.grid(linestyle = '--', axis='both')
plt.tight_layout(pad = 0.1)
plt.legend(ncol = 4, borderpad = 0.2, labelspacing=0.2, borderaxespad=0.2, handletextpad=0.4, handlelength=1.2, columnspacing = 0.8, loc = "upper center", fancybox = False)
plt.show()


############ 



fig, ax1 = plt.subplots(figsize=(6, 3))

key = list(rewardD['zTT2target65'][0].keys())[0]
deft = rewardD['zTT2target65'][0][key]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
deft3 = np.array(deft3)
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
ax1.plot(deft2, deft3, color = cmap(4), alpha = 0.95, linestyle = "-.")

key = list(rewardD['gearDVFStargetUtil0.0'][0].keys())[0]
deft = rewardD['gearDVFStargetUtil0.0'][0][key]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
deft3 = np.array(deft3)
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
ax1.plot(deft2, deft3, color = cmap(8), alpha = 0.95, linestyle = (0, (1, 0.5)))

key = list(rewardD['DVFStrain11temp20'][0].keys())[0]
deft = rewardD['DVFStrain11temp20'][0][key]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
deft3 = np.array(deft3)
deft3 = (deft3 - deft3.min()) / (deft3.max() - deft3.min())
ax1.plot(deft2, deft3, color = cmap(12), alpha = 0.95)
ax1.set_ylabel("Reward")
ax1.set_xlabel("Time (s)")
ax1.grid(linestyle = '--', axis='both')
plt.tight_layout(pad = 0.1)
plt.show()


############ 

fig, ax1 = plt.subplots(figsize=(6, 3))


deft = bigD['defaulttemp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000000)
ax1.plot(deft2, deft3, color = cmap(0), alpha = 0.95, label = "Deft", linestyle = "--")

deft = bigD['zTT2target65'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 1000000)
ax1.plot(deft2, deft3, color = cmap(4), alpha = 0.95, label = "zTT", linestyle = "-.")

deft = bigD['gearDVFStargetUtil0.0'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 1000000)
ax1.plot(deft2, deft3, color = cmap(8), alpha = 0.95, label = "Gear", linestyle = (0, (1, 0.5)))

deft = bigD['DVFStrain11temp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 1000000)
ax1.plot(deft2, deft3, color = cmap(12), alpha = 0.95, label = "Ear")
ax1.set_ylabel("Big frequency")
ax1.set_xlabel("Time (s)")
ax1.grid(linestyle = '--', axis='both')
# ax1.legend(ncol = 2, borderpad = 0.2, labelspacing=0.2, borderaxespad=0, handletextpad=0.4, handlelength=1, columnspacing = 1, loc = "upper right", fancybox = False, framealpha = 0.5)
plt.tight_layout(pad = 0.1)
plt.show()

############ 

fig, ax1 = plt.subplots(figsize=(6, 3))


deft = utilD['defaulttemp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 * 100)
ax1.plot(deft2, deft3, color = cmap(0), alpha = 0.95, linestyle = "--")

deft = utilD['zTT2target65'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 * 100)
ax1.plot(deft2, deft3, color = cmap(4), alpha = 0.95, linestyle = "-.")

deft = utilD['gearDVFStargetUtil0.0'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 * 100)
ax1.plot(deft2, deft3, color = cmap(8), alpha = 0.95, linestyle = (0, (1, 0.5)))

deft = utilD['DVFStrain11temp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 * 100)
ax1.plot(deft2, deft3, color = cmap(12), alpha = 0.95)
ax1.set_ylabel("Big utilization")
ax1.set_xlabel("Time (s)")
ax1.grid(linestyle = '--', axis='both')
plt.tight_layout(pad = 0.1)
plt.show()

############ 


############ 













fig, ax1 = plt.subplots(figsize=(6, 3))


deft = bigD['defaulttemp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000000)
ax1.plot(deft2, deft3, color = cmap(0), alpha = 0.95, label = "Deft", linestyle = "--")

deft = bigD['zTT2target65'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 1000000)
ax1.plot(deft2, deft3, color = cmap(4), alpha = 0.95, label = "zTT", linestyle = "-.")

deft = bigD['gearDVFStargetUtil0.0'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 1000000)
ax1.plot(deft2, deft3, color = cmap(8), alpha = 0.95, label = "Gear", linestyle = (0, (1, 0.5)))

deft = bigD['DVFStrain11temp20'][0]
deft2, deft3, firstTime = [], [], deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 / 1000000)
ax1.plot(deft2, deft3, color = cmap(12), alpha = 0.95, label = "Ear", linestyle = "-")
ax1.set_ylabel("Big freq.")
ax1.set_xlabel("Time (s)")
ax1.grid(linestyle = '--', axis='both')
# ax1.legend(ncol = 2, borderpad = 0.2, labelspacing=0.2, borderaxespad=0, handletextpad=0.4, handlelength=1, columnspacing = 1, loc = "upper right", fancybox = False, framealpha = 0.5)
plt.tight_layout(pad = 0.1)
plt.show()