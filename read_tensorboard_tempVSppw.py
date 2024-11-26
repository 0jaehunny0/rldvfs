from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os

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
tempDict2 = {}
tempDict3 = {}
for subdir, dirs, files in os.walk("selected/tempVSppw"):
    a = subdir.split("/")

    

    if len(a) > 2:
        mykey = a[2].split("_")[2] + a[2].split("_")[4]
        if mykey not in mydict:
            mydict[mykey] = []
            ppwDict[mykey] = []
            myfpsDict[mykey] = []
            tempDict[mykey] = []
            tempDict2[mykey] = []
            tempDict3[mykey] = []

for subdir, dirs, files in os.walk("selected/tempVSppw"):
    # print(subdir)
    for file in files:

        a = subdir.split("/")
        if len(a) > 2:
            mykey = a[2].split("_")[2] + a[2].split("_")[4]

        filepath = subdir + os.sep + file
        if "events" in filepath:
            x = parse_tensorboard(filepath, scalars)
            print(subdir, x["perf/ppw"][-50:]["value"].mean())
            # print(subdir, x["perf/fps"][-10:]["value"].mean())

            mydict[mykey].append(x["perf/ppw"][:40]["value"].mean())
            
            myfpsDict[mykey].append(x["perf/fps"][:40]["value"].mean())
            tempDict[mykey].append(x["perf/power"][:40]["value"].mean())

            # tempDict[mykey].append(x["perf/power"][:24]["value"].iloc[-1])

            xxx = x["temp/little"][:24]["value"].iloc[-1] + x["temp/big"][:24]["value"].iloc[-1] + x["temp/mid"][:24]["value"].iloc[-1]
            tempDict2[mykey].append(xxx/3)

            xxx = x["temp/gpu"][:24]["value"].iloc[-1]
            tempDict3[mykey].append(xxx)
            # mydict[mykey] = 

# ppwDict = {}
# for k,v in mydict.items():
#     # v is the list of grades for student k
#     ppwDict[k] = sum(v)/ float(len(v))

# fpsDict = {}
# for k,v in myfpsDict.items():
#     # v is the list of grades for student k
#     fpsDict[k] = sum(v)/ float(len(v))

# temps = {}
# for k,v in tempDict.items():
#     # v is the list of grades for student k
#     temps[k] = sum(v)/ float(len(v))

# import PyQt5
# matplotlib.use('Qt5Agg')

ppwLi = [mydict["freqFixed20.0"][0], mydict["freqFixed30.0"][0], mydict["freqFixed40.0"][0]]
fpsLi = [myfpsDict["freqFixed20.0"][0], myfpsDict["freqFixed30.0"][0], myfpsDict["freqFixed40.0"][0]]
powerLi = [tempDict["freqFixed20.0"][0], tempDict["freqFixed30.0"][0], tempDict["freqFixed40.0"][0]]
cpuLi = [tempDict2["freqFixed20.0"][0], tempDict2["freqFixed30.0"][0], tempDict2["freqFixed40.0"][0]]
gpuLi = [tempDict3["freqFixed20.0"][0], tempDict3["freqFixed30.0"][0], tempDict3["freqFixed40.0"][0]]





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




df = pd.DataFrame()

df["CPU"] = cpuLi
df["GPU"] = gpuLi




# ax = df.plot(kind = "bar", secondary_y = "Power", figsize=(8.5, 4), rot=0)
ax = df.plot(kind = "bar", secondary_y = "GPU", figsize=(4.8, 3.3), rot=0)


handles, labels = ax.get_legend_handles_labels()
handles_right, labels_right = ax.right_ax.get_legend_handles_labels()
labels_right = ["GPU"]
ax.legend(handles + handles_right, labels + labels_right, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False, borderpad=0.1, columnspacing=0.5)
ax.set_xticklabels(["10°C", "20°C", "30°C"])
ax.set_xlabel("Ambient temperature")
ax.set_ylabel("CPU temperature (°C)")
ax.get_ylim()
right_ax_label = ax.right_ax.set_ylabel("GPU temperature (°C)")
ax.right_ax.set_ylim(ax.get_ylim())
# right_ax_label.set_rotation(270) 

ax.grid(linestyle = "--", axis="y")
ax.axes.set_axisbelow(True)
ax.set_axisbelow(True)
ax.right_ax.set_axisbelow(True)

plt.tight_layout(pad=0.1)
plt.show()










df = pd.DataFrame()

df["Frame rates"] = fpsLi
df["Power"] = np.array(powerLi)/1000




# ax = df.plot(kind = "bar", secondary_y = "Power", figsize=(8.5, 4), rot=0)
ax = df.plot(kind = "bar", secondary_y = "Power", figsize=(6, 3.6), rot=0)


handles, labels = ax.get_legend_handles_labels()
handles_right, labels_right = ax.right_ax.get_legend_handles_labels()
labels_right = ["Power"]
ax.legend(handles + handles_right, labels + labels_right, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False, borderpad=0.1)

ax.set_xticklabels(["10°C", "20°C", "30°C"])
ax.set_xlabel("Ambient temperature")
ax.set_ylabel("Frame rates (fps)")
right_ax_label = ax.right_ax.set_ylabel("Power consumption (W)")
# right_ax_label.set_rotation(270) 

plt.tight_layout(pad=0.5)
plt.show()

exit(1)

print(1 - ppwDict['default'] / ppwDict['DVFStrain6'])
print(1 - ppwDict['zTT'] / ppwDict['DVFStrain6'])
print(1 - ppwDict['gearDVFS'] / ppwDict['DVFStrain6'])

print(1 - fpsDict['default'] / fpsDict['DVFStrain6'])
print(1 - fpsDict['zTT'] / fpsDict['DVFStrain6'])
print(1 - fpsDict['gearDVFS'] / fpsDict['DVFStrain6'])


print(1 - temps['default'] / temps['DVFStrain6'])
print(1 - temps['zTT'] / temps['DVFStrain6'])
print(1 - temps['gearDVFS'] / temps['DVFStrain6'])


a=1