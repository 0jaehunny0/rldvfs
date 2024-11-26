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
    "cstate/big",
    "cstate/mid",
    "cstate/little",
    "cstate/gpu"
]

mydict = {}
ppwDict = {}
myfpsDict = {}
tempDict = {}
for subdir, dirs, files in os.walk("selected/thermalcondition"):
    a = subdir.split("/")


    if len(a) > 2:
        if len(a) > 2:
            if "gear" in a[2].split("_")[2]:
                mykey = a[2].split("_")[2] + a[2].split("_")[-3]
            else:
                mykey = a[2].split("_")[2] + a[2].split("_")[-1]

        if mykey not in mydict:
            mydict[mykey] = []
            ppwDict[mykey] = []
            myfpsDict[mykey] = []
            tempDict[mykey] = []    


for subdir, dirs, files in os.walk("selected/thermalcondition"):
    # print(subdir)
    for file in files:

        a = subdir.split("/")
        if len(a) > 2:
            if "gear" in a[2].split("_")[2]:
                mykey = a[2].split("_")[2] + a[2].split("_")[-3]
            else:
                mykey = a[2].split("_")[2] + a[2].split("_")[-1]

        filepath = subdir + os.sep + file
        if "events" in filepath:
            x = parse_tensorboard(filepath, scalars)
            print(subdir, x["perf/ppw"][-50:]["value"].mean())
            # print(subdir, x["perf/fps"][-10:]["value"].mean())

            mydict[mykey].append(x["perf/ppw"][:]["value"].mean())
            
            myfpsDict[mykey].append(x["perf/fps"][:]["value"].mean())

            cstate = x["cstate/big"]
            cstate["value"] = cstate["value"] + x["cstate/mid"]["value"]
            cstate["value"] = cstate["value"] + x["cstate/little"]["value"]
            cstate["value"] = cstate["value"] + x["cstate/gpu"]["value"]

            tempDict[mykey].append(cstate)
            # mydict[mykey] = 

zttppw = np.array([mydict["zTT2target55"], mydict["zTT2target60"], mydict["zTT2target65"],mydict["zTT2target70"], mydict["zTT2target75"]])
zttfps = np.array([myfpsDict["zTT2target55"], myfpsDict["zTT2target60"], myfpsDict["zTT2target65"],myfpsDict["zTT2target70"], myfpsDict["zTT2target75"]])

zttppw = np.array([mydict["zTT2target55"], mydict["zTT2target65"],mydict["zTT2target75"]])
zttfps = np.array([myfpsDict["zTT2target55"], myfpsDict["zTT2target65"],myfpsDict["zTT2target75"]])


gearppw = np.array([mydict["gearDVFStarget40"], mydict["gearDVFStarget50"],mydict["gearDVFStarget60"]])
gearfps = np.array([myfpsDict["gearDVFStarget40"], myfpsDict["gearDVFStarget50"],myfpsDict["gearDVFStarget60"]])


print((zttppw.T)[0])
print((zttfps.T)[0])
print((gearppw.T)[0])
print((gearfps.T)[0])

mydict.keys()

ours =  tempDict['zTT2target70'][0]
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


deft = tempDict['defaulttemp20'][0]
limit = tempDict["limitDefaultgpu400000"][0]

deft = myfpsDict['defaulttemp20'][0]
limit = myfpsDict["limitDefaultgpu400000"][0]



limit2 = []
limit3 = []

firstTime = limit["wall_time"][0]
for i in range(len(limit["step"])):
    if i %4 == 3:
        limit2.append(limit["wall_time"][i] - firstTime)
        limit3.append(sum(limit["value"][i-3:i+1]) / 4)


deft2 = []
deft3 = []

firstTime = deft["wall_time"][0]
for i in range(len(deft["step"])):
    if i %2 == 1:
        deft2.append(deft["wall_time"][i] - firstTime)
        deft3.append(sum(deft["value"][i-1:i+1]) / 2)

plt.figure(figsize=(8,4.5))
plt.plot(limit2, limit3, label = "limit")
plt.plot(deft2, deft3, label = "default")
plt.legend()
plt.show()

