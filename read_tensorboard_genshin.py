from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import numpy as np
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
    "perf/fps",
    "perf/ppw",
    "perf/power",
    "temp/gpu"
]

mydict = {}
ppwDict = {}
myfpsDict = {}
tempDict = {}
for subdir, dirs, files in os.walk("selected/genshin"):
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


for subdir, dirs, files in os.walk("selected/genshin"):
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
            print(subdir, x["perf/ppw"][-50:]["value"].mean())
            # print(subdir, x["perf/fps"][-10:]["value"].mean())

            mydict[mykey].append(x["perf/ppw"][:]["value"].mean())
            
            myfpsDict[mykey].append(x["perf/fps"][:]["value"].mean())
            tempDict[mykey].append(x["temp/gpu"][:]["value"].mean())
            # mydict[mykey] = 

ppw20 = np.array([mydict["def30.0"], mydict["zTT30.0"], mydict["gea30.0"], mydict["DVF30.0"]]).flatten()

fps20 = np.array([myfpsDict["def30.0"], myfpsDict["zTT30.0"], myfpsDict["gea30.0"], myfpsDict["DVF30.0"]]).flatten()

print("ppw20", ppw20)
print("fps20", fps20)

print(ppw20)
print(fps20)


df10 = pd.DataFrame()
df20 = pd.DataFrame()
df30 = pd.DataFrame()

df10["ppw"] = np.array(ppw10).flatten()
df10["fps"] = np.array(fps10).flatten()
df10.index = ["default", "zTT", "gearDVFS", "ours"]

df10.plot()
plt.show()

ppwDict = {}
for k,v in mydict.items():
    # v is the list of grades for student k
    ppwDict[k] = sum(v)/ float(len(v))

fpsDict = {}
for k,v in myfpsDict.items():
    # v is the list of grades for student k
    fpsDict[k] = sum(v)/ float(len(v))

temps = {}
for k,v in tempDict.items():
    # v is the list of grades for student k
    temps[k] = sum(v)/ float(len(v))

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