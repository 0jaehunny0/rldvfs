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
    "perf/power"
]

mydict = {}

ppwDict = {}
fpsDict = {}
for subdir, dirs, files in os.walk("test4"):
    a = subdir.split("/")

    

    if len(a) > 1:
        mykey = a[1].split("_")[0]
        if mykey not in mydict:
            mydict[mykey] = []
            ppwDict[mykey] = []
            fpsDict[mykey] = []
for subdir, dirs, files in os.walk("test4"):
    # print(subdir)
    for file in files:

        a = subdir.split("/")
        if len(a) > 1:
            mykey = a[1].split("_")[0]

        filepath = subdir + os.sep + file
        if "events" in filepath:
            x = parse_tensorboard(filepath, scalars)
            print(subdir, x["perf/ppw"][-50:]["value"].mean())
            # print(subdir, x["perf/fps"][-10:]["value"].mean())

            mydict[mykey].append(x["perf/ppw"][-50:]["value"].mean())
            
            fpsDict[mykey].append(x["perf/fps"][-50:]["value"].mean())
            # mydict[mykey] = 

avgDict = {}
for k,v in mydict.items():
    # v is the list of grades for student k
    avgDict[k] = sum(v)/ float(len(v))

avgDict2 = {}
for k,v in fpsDict.items():
    # v is the list of grades for student k
    avgDict2[k] = sum(v)/ float(len(v))

a=1