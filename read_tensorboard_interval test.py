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
    "temp/gpu"
]

mydict = {}
ppwDict = {}
myfpsDict = {}
tempDict = {}

custompath = "runs/interval test/starrail"
custompath = "runs/interval test/aquarium"

for subdir, dirs, files in os.walk(custompath):
    a = subdir.split("/")

    

    if len(a) > 3:
        mykey = a[-1].split("__")[0]
        if mykey not in mydict:
            mydict[mykey] = []
            ppwDict[mykey] = []
            myfpsDict[mykey] = []
            tempDict[mykey] = []

for subdir, dirs, files in os.walk(custompath):
    # print(subdir)
    for file in files:

        a = subdir.split("/")
        if len(a) > 3:
            mykey = a[-1].split("__")[0]

        filepath = subdir + os.sep + file
        if "events" in filepath:
            x = parse_tensorboard(filepath, scalars)
            print(subdir, x["perf/ppw"][-50:]["value"].mean())

            # mydict[mykey].append(x["perf/ppw"][-30:]["value"].mean()) 
            # myfpsDict[mykey].append(x["perf/fps"][-30:]["value"].mean())
            # tempDict[mykey].append(x["temp/gpu"][-30:]["value"].mean())

            mydict[mykey].append(x["perf/ppw"][71:128]["value"].mean()) 
            myfpsDict[mykey].append(x["perf/fps"][71:128]["value"].mean())
            tempDict[mykey].append(x["temp/gpu"][71:128]["value"].mean())



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

a=1