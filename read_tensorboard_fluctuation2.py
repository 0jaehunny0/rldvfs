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
    "util/gu",
]

mydict = {}
ppwDict = {}
myfpsDict = {}
tempDict = {}
batteryDict = {}
powerDict = {}
cpuDict = {}
cpuDict1 = {}
cpuDict2 = {}
cpuDict3 = {}

bigD = {}
midD = {}
litD = {}
gpuD = {}
utilD = {}

for subdir, dirs, files in os.walk("selected/fluctuation2"):
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
            cpuDict1[mykey] = []   
            cpuDict2[mykey] = []   
            cpuDict3[mykey] = []   
                        
            bigD[mykey] = []    
            midD[mykey] = []    
            litD[mykey] = []    
            gpuD[mykey] = []    
            utilD[mykey] = []    
 


for subdir, dirs, files in os.walk("selected/fluctuation2"):
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

            cstate = x["cstate/gpu"]
            # cstate["value"] = cstate["value"] + x["cstate/mid"]["value"]
            # cstate["value"] = cstate["value"] + x["cstate/little"]["value"]
            # cstate["value"] = cstate["value"] + x["cstate/gpu"]["value"]

            cputemp = x["temp/gpu"]
            cputemp1 = x["temp/big"]
            cputemp2 = x["temp/mid"]
            cputemp3 = x["temp/little"]
            # cputemp["value"] = cputemp["value"] + x["temp/mid"]["value"]
            # cputemp["value"] = cputemp["value"] + x["temp/little"]["value"] + x["temp/gpu"]["value"]

            cpuDict[mykey].append(cputemp)
            cpuDict1[mykey].append(cputemp1)
            cpuDict2[mykey].append(cputemp2)
            cpuDict3[mykey].append(cputemp3)

            tempDict[mykey].append(cstate)

            bigD[mykey].append(x["freq/big"])
            midD[mykey].append(x["freq/mid"])
            litD[mykey].append(x["freq/little"])
            gpuD[mykey].append(x["freq/gpu"])
            utilD[mykey].append(x["util/gu"])


mydict.keys()





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


fig, axes = plt.subplots(2, 3, figsize=(8, 3.2))


deft = batteryDict['defaulttemp10'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3, color = "C6")

deft = batteryDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3, color = "C7")
axes[0][0].set_xticks([])


deft = batteryDict['defaulttemp30'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3, color = "C8")
axes[0][0].set_xlabel("Battery temp. (°C)")
axes[0][0].set_xticks([])



deft = cpuDict['defaulttemp10'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][1].plot(deft2, deft3, color = "C6")

deft = cpuDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][1].plot(deft2, deft3, color = "C7")
axes[0][1].set_xticks([])


deft = cpuDict['defaulttemp30'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][1].plot(deft2, deft3, color = "C8")
axes[0][1].set_xlabel("GPU temp. (°C)")
axes[0][1].set_xticks([])



deft = cpuDict1['defaulttemp10'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][0].plot(deft2, deft3, color = "C6")

deft = cpuDict1['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][0].plot(deft2, deft3, color = "C7")
axes[1][0].set_xticks([])

deft = cpuDict1['defaulttemp30'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][0].plot(deft2, deft3, color = "C8")
axes[1][0].set_xlabel("big temp. (°C)")
axes[1][0].set_xticks([])
axes[1][0].set_ylim(64.15, 78.25)





deft = cpuDict2['defaulttemp10'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][1].plot(deft2, deft3, color = "C6")

deft = cpuDict2['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][1].plot(deft2, deft3, color = "C7")
axes[1][1].set_xticks([])
axes[1][1].set_ylim(64.15, 78.25)


deft = cpuDict2['defaulttemp30'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][1].plot(deft2, deft3, color = "C8")
axes[1][1].set_xlabel("mid temp. (°C)")
axes[1][1].set_xticks([])




deft = cpuDict3['defaulttemp10'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][2].plot(deft2, deft3, color = "C6")

deft = cpuDict3['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][2].plot(deft2, deft3, color = "C7")
axes[1][2].set_xticks([])

deft = cpuDict3['defaulttemp30'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][2].plot(deft2, deft3, color = "C8")
axes[1][2].set_xlabel("little temp. (°C)")
axes[1][2].set_xticks([])
axes[1][2].set_ylim(64.15, 78.25)





axes[0][2].axis('off') 

axes[0][0].grid(axis='y',linestyle='--')  
axes[0][0].set_axisbelow(True)

axes[0][1].grid(axis='y',linestyle='--')  
axes[0][1].set_axisbelow(True)

axes[1][0].grid(axis='y',linestyle='--')  
axes[1][0].set_axisbelow(True)


axes[1][1].grid(axis='y',linestyle='--')  
axes[1][1].set_axisbelow(True)


axes[1][2].grid(axis='y',linestyle='--')  
axes[1][2].set_axisbelow(True)




axes[0][0].set_yticks([25,35,45])


# fig.legend(["Default", "Limit"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, fontsize=22)
fig.legend(["10°C", "20°C", "30°C"], loc='upper right', ncol=1, bbox_to_anchor=(0.965, 1.03),frameon=True, fontsize=22, labelspacing=0.4)
plt.tight_layout(pad = 0.1, rect = (0,0,1,1))

plt.show()


######################





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


fig, axes = plt.subplots(2, 2, figsize=(6, 3.2))


deft = batteryDict['defaulttemp10'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3, color = "C6")

deft = batteryDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3, color = "C7")
axes[0][0].set_xticks([])


deft = batteryDict['defaulttemp30'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3, color = "C8")
axes[0][0].set_xlabel("Battery temp. (°C)")
axes[0][0].set_xticks([])




deft = gpuD['defaulttemp10'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000000)
axes[1][0].plot(deft2, deft3, color = "C6")

deft = gpuD['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000000)
axes[1][0].plot(deft2, deft3, color = "C7")
axes[1][0].set_xticks([])

deft = gpuD['defaulttemp30'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000000)
axes[1][0].plot(deft2, deft3, color = "C8")
axes[1][0].set_xlabel("GPU freq. (Ghz)")
axes[1][0].set_xticks([])




deft = cpuDict['defaulttemp10'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][1].plot(deft2, deft3, color = "C6")

deft = cpuDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][1].plot(deft2, deft3, color = "C7")
axes[1][1].set_xticks([])


deft = cpuDict['defaulttemp30'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][1].plot(deft2, deft3, color = "C8")
axes[1][1].set_xlabel("GPU temp. (°C)")
axes[1][1].set_xticks([])

axes[0][1].axis('off') 

# fig.legend(["Default", "Limit"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, fontsize=22)
fig.legend(["10°C", "20°C", "30°C"], loc='upper right', ncol=1, bbox_to_anchor=(0.95, 1.03),frameon=True, fontsize=22)
plt.tight_layout(pad = 0.1, rect = (0,0,1,1))

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


deft = batteryDict['defaulttemp10'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3)

deft = batteryDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3)
axes[0][0].set_xlabel("Frame rates (fps)")
axes[0][0].set_xticks([])


deft = batteryDict['defaulttemp30'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][0].plot(deft2, deft3)
axes[0][0].set_xlabel("Battery temperature (°C)")
axes[0][0].set_xticks([])





deft = tempDict['defaulttemp10'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][1].plot(deft2, deft3)

deft = tempDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][1].plot(deft2, deft3)
axes[0][1].set_xlabel("Power consumption (W)")
axes[0][1].set_xticks([])


deft = tempDict['defaulttemp30'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0][1].plot(deft2, deft3)
axes[0][1].set_xlabel("Total cooling states")
axes[0][1].set_xticks([])




deft = gpuD['defaulttemp10'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000000)
axes[1][0].plot(deft2, deft3)

deft = gpuD['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000000)
axes[1][0].plot(deft2, deft3)
axes[1][0].set_xlabel("Total cooling states")
axes[1][0].set_xticks([])

deft = gpuD['defaulttemp30'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 / 1000000)
axes[1][0].plot(deft2, deft3)
axes[1][0].set_xlabel("GPU frequency (Ghz)")
axes[1][0].set_xticks([])




deft = cpuDict['defaulttemp10'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][1].plot(deft2, deft3)

deft = cpuDict['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][1].plot(deft2, deft3)
axes[1][1].set_xlabel("CPU temperature (°C)")
axes[1][1].set_xticks([])


deft = cpuDict['defaulttemp30'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[1][1].plot(deft2, deft3)
axes[1][1].set_xlabel("GPU temperature (°C)")
axes[1][1].set_xticks([])

# fig.legend(["Default", "Limit"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, fontsize=22)
fig.legend(["10°C", "20°C", "30°C"], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, fontsize=22)
plt.tight_layout(pad = 0.5, rect = (0,0,1,0.95))

plt.show()










############ figure 4




fig, axes = plt.subplots(1, 2, figsize=(10, 4))

deft = bigD['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)
axes[0].plot(deft2, deft3)

deft = bigD['zTT2target65'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
axes[0].plot(deft2, deft3)
axes[0].set_xlabel("Frame rates (fps)")
axes[0].set_xticks([])

deft = bigD['gearDVFStargetUtil0.0'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
axes[0].plot(deft2, deft3)
axes[0].set_xlabel("Frame rates (fps)")
axes[0].set_xticks([])

deft = bigD['DVFStrain11temp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4)
axes[0].plot(deft2, deft3)
axes[0].set_xlabel("Frame rates (fps)")
axes[0].set_xticks([])





deft = utilD['defaulttemp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2 *100)
axes[1].plot(deft2, deft3)

deft = utilD['zTT2target65'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 *100)
axes[1].plot(deft2, deft3)
axes[1].set_xlabel("Power consumption (W)")
axes[1].set_xticks([])

deft = utilD['gearDVFStargetUtil0.0'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 *100)
axes[1].plot(deft2, deft3)
axes[1].set_xlabel("Power consumption (W)")
axes[1].set_xticks([])

deft = utilD['DVFStrain11temp20'][0]
deft2, deft3 = [], []
firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %4 == 3:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-3:i+1]) / 4 *100)
axes[1].plot(deft2, deft3)
axes[1].set_xlabel("Power consumption (W)")
axes[1].set_xticks([])

fig.legend(["Default", "zTT", "gear", "ear"], loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1)
plt.tight_layout(pad = 0.5, rect = (0,0,1,0.95))

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