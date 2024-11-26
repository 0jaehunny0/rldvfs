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
    "temp/gpu"
]


mydict = {}
ppwDict = {}
myfpsDict = {}
tempDict = {}
powerDict = {}
for subdir, dirs, files in os.walk("selected/zTTcmp"):
    a = subdir.split("/")


    if len(a) > 2:
        if len(a) > 2:
            mykey = a[-1]

        if mykey not in mydict:
            mydict[mykey] = []
            ppwDict[mykey] = []
            myfpsDict[mykey] = []
            tempDict[mykey] = []    
            powerDict[mykey] = []    


for subdir, dirs, files in os.walk("selected/zTTcmp"):
    # print(subdir)
    for file in files:

        a = subdir.split("/")
        if len(a) > 2:
            mykey = a[-1]
        filepath = subdir + os.sep + file
        if "events" in filepath:
            x = parse_tensorboard(filepath, scalars)
            print(subdir, x["perf/ppw"][-50:]["value"].mean())
            # print(subdir, x["perf/fps"][-10:]["value"].mean())

            mydict[mykey].append(x["perf/ppw"][:]["value"].mean())
            powerDict[mykey].append(x["perf/power"][:]["value"].mean())
            myfpsDict[mykey].append(x["perf/fps"][:]["value"].mean())
            tempDict[mykey].append(x["temp/gpu"][:]["value"].mean())
            # mydict[mykey] = 

import matplotlib

myfpsDict

myfpsDict[list(myfpsDict.keys())[0]]

deft = list(myfpsDict.keys())[1]

zTT65 = list(myfpsDict.keys())[0]
zTT55 = list(myfpsDict.keys())[2]
zTT75 = list(myfpsDict.keys())[3]

myfpsDict[zTT75][0] / myfpsDict[deft][0]
mydict[zTT75][0] / mydict[deft][0]


myfpsDict[zTT75][0] / myfpsDict[zTT65][0]
mydict[zTT75][0] / mydict[zTT65][0]


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
import matplotlib.patches as mpatches


########################
import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graphs
categories = ['10°C', '20°C', '30°C']  # Y-axis labels
values1 = [
    [myfpsDict["def20.0"][0].mean(), myfpsDict["def30.0"][0].mean(), myfpsDict["def40.0"][0].mean()],  # Group 1
    [myfpsDict["zTT20.0"][0].mean(), myfpsDict["zTT30.0"][0].mean(), myfpsDict["zTT40.0"][0].mean()],  # Group 1
    [myfpsDict["gea20.0"][0].mean(), myfpsDict["gea30.0"][0].mean(), myfpsDict["gea40.0"][0].mean()],  # Group 1
 ]
values2 = [
    [mydict["def20.0"][0].mean(), mydict["def30.0"][0].mean(), mydict["def40.0"][0].mean()],  # Group 1
    [mydict["zTT20.0"][0].mean(), mydict["zTT30.0"][0].mean(), mydict["zTT40.0"][0].mean()],  # Group 1
    [mydict["gea20.0"][0].mean(), mydict["gea30.0"][0].mean(), mydict["gea40.0"][0].mean()],  # Group 1
]

labelLi = ["def", "zTT", "gear"]
values1 = np.array(values1).T
values2 = np.array(values2).T

values2 *= 1000

# Number of groups per category
num_groups = len(values1[0])
bar_width = 0.20  # Width of individual bars
y_positions = np.arange(len(categories))  # Position for each category

# Create figure and two side-by-side subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharey=True)

# First subplot on the left
for i in range(num_groups):
    ax1.barh(y_positions + i * bar_width, [row[i] for row in values1], 
             height=bar_width, label=labelLi[i], color = cmap(i*4))
#ax1.invert_xaxis()  # Invert to keep bars directed toward the shared y-axis
ax1.set_yticks(y_positions + bar_width)  # Set y-ticks for categories
#ax1.set_yticklabels(["", "", ""])  # Empty labels for ax1
#ax1.yaxis.tick_right()  # Move y-ticks to the right side of ax1
ax1.set_xlabel("Frame rates (fps)")
#ax1.legend()


# Second subplot on the right
for i in range(num_groups):
    ax2.barh(y_positions + i * bar_width, [row[i] for row in values2], 
             height=bar_width, label=labelLi[i], color = cmap(i*4))
ax2.set_yticks(y_positions + bar_width)  # Align y-ticks
ax2.set_yticklabels(categories)  # Set categories on the shared middle y-axis
ax2.set_xlabel("Power efficiency (PPW)")



red_patch = mpatches.Patch(color=cmap(0), label='Default governor')
fig.legend(handles=[red_patch], loc='upper center', ncol=2, bbox_to_anchor=(0.42, 1.035),frameon=False, borderpad=0.1, columnspacing = 4.0)

r_patch = mpatches.Patch(color=cmap(4), label='zTT')
x_patch = mpatches.Patch(color=cmap(8), label='GearDVFS')
# fig.legend(handles=[r_patch, x_patch], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, columnspacing = 7.0)

# Adjust layout for the correct appearance
plt.tight_layout(pad=0.5, rect=[0, 0, 1, 0.95])  # Leave space at the top for legend


# Add space between y-axis labels and the bars in ax2
#ax1.yaxis.set_tick_params(pad=9)  # Increase padding between labels and bars

plt.show()

#####################################



########################
import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graphs
categories = ['10°C', '20°C', '30°C']  # X-axis labels
values1 = [
    [myfpsDict["def20.0"][0].mean(), myfpsDict["def30.0"][0].mean(), myfpsDict["def40.0"][0].mean()],
    [myfpsDict["zTT20.0"][0].mean(), myfpsDict["zTT30.0"][0].mean(), myfpsDict["zTT40.0"][0].mean()],  # Group 1
    [myfpsDict["gea20.0"][0].mean(), myfpsDict["gea30.0"][0].mean(), myfpsDict["gea40.0"][0].mean()],  # Group 1

]
values2 = [
    [mydict["def20.0"][0].mean(), mydict["def30.0"][0].mean(), mydict["def40.0"][0].mean()],
    [mydict["zTT20.0"][0].mean(), mydict["zTT30.0"][0].mean(), mydict["zTT40.0"][0].mean()],  # Group 1
    [mydict["gea20.0"][0].mean(), mydict["gea30.0"][0].mean(), mydict["gea40.0"][0].mean()],  # Group 1
]


values1 = np.array(values1)
values2 = np.array(values2)

v1 = np.array([myfpsDict["def20.0"][0].mean(), myfpsDict["def30.0"][0].mean(), myfpsDict["def40.0"][0].mean()])
v2 = np.array([mydict["def20.0"][0].mean(), mydict["def30.0"][0].mean(), mydict["def40.0"][0].mean()])

# values1 = 100 * (values1 / v1) - 100
# values2 = 100 * (values2 / v2) - 100

labelLi = ["zTT", "gear", "ours"]
values1 = np.array(values1).T
values2 = np.array(values2).T

# Number of groups per category
num_groups = len(values1[0])
bar_width = 0.20  # Width of individual bars
x_positions = np.arange(len(categories))  # Position for each category

# Create figure and two side-by-side subplots
fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(4, 6))

# Second subplot (ax2)
for i in range(num_groups):
    ax2.bar(x_positions + i * bar_width, [row[i] for row in values2], 
            width=bar_width, label=labelLi[i])
ax2.set_xticks([]) 
ax2.set_ylabel("PPW change (%)")
ax2.legend(borderpad = 0.3, labelspacing=0.2, borderaxespad=0.3, handletextpad=0.4, handlelength=1.5)
ax2.grid(axis='y',linestyle='--')  # Add grid to the second subplot
ax2.set_axisbelow(True)

# First subplot (ax1)
for i in range(num_groups):
    ax1.bar(x_positions + i * bar_width, [row[i] for row in values1], 
            width=bar_width, label=labelLi[i])
ax1.set_xticks(x_positions + bar_width)  # Set x-ticks for categories
ax1.set_xticklabels(categories)  # Set categories on the shared middle x-axis
ax1.grid(axis='y',linestyle='--')  # Add grid to the first subplot
ax1.set_axisbelow(True)
ax1.set_ylabel("FPS change (%)")
# Adjust layout
plt.tight_layout(pad=0.5)

print(ax2.get_yticklabels())

# lab = ax2.get_yticklabels()
# lab2 = []
# for i in lab:
#     if i._y > 0:
#         lab2.append("+"+str(int(i._y)))
#     else:
#         lab2.append(str(int(i._y)))

# ax2.set_yticklabels(lab2)
plt.show()

########################
import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graphs
categories = ['10°C', '20°C', '30°C']  # X-axis labels
values1 = [
    [myfpsDict["zTT20.0"][0].mean(), myfpsDict["zTT30.0"][0].mean(), myfpsDict["zTT40.0"][0].mean()],  # Group 1
    [myfpsDict["gea20.0"][0].mean(), myfpsDict["gea30.0"][0].mean(), myfpsDict["gea40.0"][0].mean()],  # Group 1
    [myfpsDict["DVF20.0"][0].mean(), myfpsDict["DVF30.0"][0].mean(), myfpsDict["DVF40.0"][0].mean()],  # Group 1
]
values2 = [
    [mydict["zTT20.0"][0].mean(), mydict["zTT30.0"][0].mean(), mydict["zTT40.0"][0].mean()],  # Group 1
    [mydict["gea20.0"][0].mean(), mydict["gea30.0"][0].mean(), mydict["gea40.0"][0].mean()],  # Group 1
    [mydict["DVF20.0"][0].mean(), mydict["DVF30.0"][0].mean(), mydict["DVF40.0"][0].mean()],  # Group 1
]


values1 = np.array(values1)
values2 = np.array(values2)

v1 = np.array([myfpsDict["def20.0"][0].mean(), myfpsDict["def30.0"][0].mean(), myfpsDict["def40.0"][0].mean()])
v2 = np.array([mydict["def20.0"][0].mean(), mydict["def30.0"][0].mean(), mydict["def40.0"][0].mean()])

values1 = 100 * (values1 / v1) - 100
values2 = 100 * (values2 / v2) - 100

labelLi = ["zTT", "gear", "ours"]
values1 = np.array(values1).T
values2 = np.array(values2).T

# Number of groups per category
num_groups = len(values1[0])
bar_width = 0.20  # Width of individual bars
x_positions = np.arange(len(categories))  # Position for each category

# Create figure and two side-by-side subplots
fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(4, 6))

# Second subplot (ax2)
for i in range(num_groups):
    ax2.bar(x_positions + i * bar_width, [row[i] for row in values2], 
            width=bar_width, label=labelLi[i])
ax2.set_xticks([]) 
ax2.set_ylabel("PPW change (%)")
ax2.legend(borderpad = 0.3, labelspacing=0.2, borderaxespad=0.3, handletextpad=0.4, handlelength=1.5)
ax2.grid(axis='y',linestyle='--')  # Add grid to the second subplot
ax2.set_axisbelow(True)

# First subplot (ax1)
for i in range(num_groups):
    ax1.bar(x_positions + i * bar_width, [row[i] for row in values1], 
            width=bar_width, label=labelLi[i])
ax1.set_xticks(x_positions + bar_width)  # Set x-ticks for categories
ax1.set_xticklabels(categories)  # Set categories on the shared middle x-axis
ax1.grid(axis='y',linestyle='--')  # Add grid to the first subplot
ax1.set_axisbelow(True)
ax1.set_ylabel("FPS change (%)")
# Adjust layout
plt.tight_layout(pad=0.5)

print(ax2.get_yticklabels())

lab = ax2.get_yticklabels()
lab2 = []
for i in lab:
    if i._y > 0:
        lab2.append("+"+str(int(i._y)))
    else:
        lab2.append(str(int(i._y)))

ax2.set_yticklabels(lab2)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data for the box plots
categories = ['10°C', '20°C', '30°C']  # Y-axis labels
# Three groups of values per category
values1 = [
    [myfpsDict["def20.0"][0].values, myfpsDict["def30.0"][0].values, myfpsDict["def40.0"][0].values],  # Group 1
    [myfpsDict["zTT20.0"][0].values, myfpsDict["zTT30.0"][0].values, myfpsDict["zTT40.0"][0].values],  # Group 1
    [myfpsDict["gea20.0"][0].values, myfpsDict["gea30.0"][0].values, myfpsDict["gea40.0"][0].values],  # Group 1
    [myfpsDict["DVF20.0"][0].values, myfpsDict["DVF30.0"][0].values, myfpsDict["DVF40.0"][0].values],  # Group 1
 ]
values2 = [
    [mydict["def20.0"][0].values, mydict["def30.0"][0].values, mydict["def40.0"][0].values],  # Group 1
    [mydict["zTT20.0"][0].values, mydict["zTT30.0"][0].values, mydict["zTT40.0"][0].values],  # Group 1
    [mydict["gea20.0"][0].values, mydict["gea30.0"][0].values, mydict["gea40.0"][0].values],  # Group 1
    [mydict["DVF20.0"][0].values, mydict["DVF30.0"][0].values, mydict["DVF40.0"][0].values],  # Group 1
]


# Position for each category
y_positions = np.arange(len(categories))

# Colors for each group
colors = ['blue', 'green', 'orange', 'red']

# Create figure and two side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

# Plot three box plots for values1 on the left subplot
for i, group in enumerate(values1):
    bp = ax1.boxplot(group, vert=False, positions=y_positions + i * 0.2, widths=0.15,
                     patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.5),
                     medianprops=dict(color="black"), showfliers=False)
ax1.invert_xaxis()  # Invert to keep boxes directed toward the shared y-axis
ax1.set_yticks(y_positions + 0.2)  # Set y-ticks for categories
ax1.set_yticklabels(["", "", ""])  # Empty labels for ax1
ax1.yaxis.tick_right()  # Move y-ticks to the right side of ax1
ax1.set_xlabel("FPS")

# Plot three box plots for values2 on the right subplot
for i, group in enumerate(values2):
    bp = ax2.boxplot(group, vert=False, positions=y_positions + i * 0.2, widths=0.15,
                     patch_artist=True, boxprops=dict(facecolor=colors[i], alpha=0.5),
                     medianprops=dict(color="black"), showfliers=False)
ax2.set_yticks(y_positions + 0.2)  # Align y-ticks
ax2.set_yticklabels(categories)  # Set categories on the shared middle y-axis
ax2.set_xlabel("PPW")

# Add space between y-axis labels and the boxes in ax2
ax2.yaxis.set_tick_params(pad=15)  # Increase padding between labels and boxes

labelLi = ["def", "zTT", "gear", "ours"]

# Add legend to the figure
handles = [plt.Line2D([0], [0], color=colors[i], lw=4, label=labelLi[i]) for i in range(len(colors))]
fig.legend(handles=handles, loc='upper center', ncol=4, title="Groups")

# Adjust layout for the correct appearance
plt.tight_layout(pad=1, rect=[0, 0, 1, 0.9])  # Leave space at the top for legend
plt.show()

########################





#####################################


aaa = [myfpsDict["def20.0"][0].mean(), myfpsDict["zTT20.0"][0].mean(), myfpsDict["gea20.0"][0].mean()]
bbb = [21.237684292579765, 23.25882270783757, 23.259738350934523]

aaa = [myfpsDict["def30.0"][0].mean(), myfpsDict["zTT30.0"][0].mean(), myfpsDict["gea30.0"][0].mean()]
bbb = [powerDict["def30.0"][0].mean(), powerDict["zTT30.0"][0].mean(), powerDict["gea30.0"][0].mean()]

aaa = [myfpsDict["def40.0"][0].mean(), myfpsDict["zTT40.0"][0].mean(), myfpsDict["gea40.0"][0].mean()]
bbb = [tempDict["def40.0"][0].mean(), tempDict["zTT40.0"][0].mean(), tempDict["gea40.0"][0].mean()]
bbb = [powerDict["def40.0"][0].mean(), powerDict["zTT40.0"][0].mean(), powerDict["gea40.0"][0].mean()]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.bar(["def","zTT","gear"], aaa)
ax1.set_xlabel("app 1 10.0")
# ax2.bar(["def","zTT","gear"], np.array(bbb)/1000)
ax2.bar(["def","zTT","gear"], np.array(bbb))
ax2.set_xlabel("app 2 30.0")
plt.tight_layout(pad=0.5)
plt.show()


[powerDict["def30.0"][0].mean(), powerDict["zTT30.0"][0].mean(), powerDict["gea30.0"][0].mean()]





####################################


plt.figure(figsize=(10,4))
plt.scatter(myfpsDict["def30.0"][0].values, powerDict["def30.0"][0].values, alpha=0.5, label = "default")
plt.scatter(myfpsDict["zTT30.0"][0].values, powerDict["zTT30.0"][0].values, alpha=0.5, label = "zTT")
plt.scatter(myfpsDict["gea30.0"][0].values, powerDict["gea30.0"][0].values, alpha=0.5, label = "gearDVFS")
plt.scatter(myfpsDict["DVF30.0"][0].values, powerDict["DVF30.0"][0].values, alpha=0.5, label = "ours")
plt.legend()

plt.tight_layout(pad=0.5)



plt.figure(figsize=(10,4))
plt.scatter(myfpsDict["def30.0"][0].values, mydict["def30.0"][0].values, alpha=0.5, label = "default")
plt.scatter(myfpsDict["zTT30.0"][0].values, mydict["zTT30.0"][0].values, alpha=0.5, label = "zTT")
plt.scatter(myfpsDict["gea30.0"][0].values, mydict["gea30.0"][0].values, alpha=0.5, label = "gearDVFS")
plt.scatter(myfpsDict["DVF30.0"][0].values, mydict["DVF30.0"][0].values, alpha=0.5, label = "ours")
plt.legend()

plt.tight_layout(pad=0.5)



plt.figure(figsize=(10,4))
plt.scatter(myfpsDict["def20.0"][0].values, mydict["def20.0"][0].values, alpha=0.5, label = "default")
plt.scatter(myfpsDict["zTT20.0"][0].values, mydict["zTT20.0"][0].values, alpha=0.5, label = "zTT")
plt.scatter(myfpsDict["gea20.0"][0].values, mydict["gea20.0"][0].values, alpha=0.5, label = "gearDVFS")
plt.scatter(myfpsDict["DVF20.0"][0].values, mydict["DVF20.0"][0].values, alpha=0.5, label = "ours")
plt.legend()

plt.tight_layout(pad=0.5)


#####################################


ppw10 = np.array([mydict["def20.0"], mydict["zTT20.0"], mydict["gea20.0"], mydict["DVF20.0"]]).flatten()
ppw15 = np.array([mydict["def25.0"], mydict["zTT25.0"], mydict["gea25.0"], mydict["DVF25.0"]]).flatten()
ppw20 = np.array([mydict["def30.0"], mydict["zTT30.0"], mydict["gea30.0"], mydict["DVF30.0"]]).flatten()
ppw25 = np.array([mydict["def35.0"], mydict["zTT35.0"], mydict["gea35.0"], mydict["DVF35.0"]]).flatten()
ppw30 = np.array([mydict["def40.0"], mydict["zTT40.0"], mydict["gea40.0"], mydict["DVF40.0"]]).flatten()

fps10 = np.array([myfpsDict["def20.0"], myfpsDict["zTT20.0"], myfpsDict["gea20.0"], myfpsDict["DVF20.0"]]).flatten()
fps15 = np.array([myfpsDict["def25.0"], myfpsDict["zTT25.0"], myfpsDict["gea25.0"], myfpsDict["DVF25.0"]]).flatten()
fps20 = np.array([myfpsDict["def30.0"], myfpsDict["zTT30.0"], myfpsDict["gea30.0"], myfpsDict["DVF30.0"]]).flatten()
fps25 = np.array([myfpsDict["def35.0"], myfpsDict["zTT35.0"], myfpsDict["gea35.0"], myfpsDict["DVF35.0"]]).flatten()
fps30 = np.array([myfpsDict["def40.0"], myfpsDict["zTT40.0"], myfpsDict["gea40.0"], myfpsDict["DVF40.0"]]).flatten()


df = pd.DataFrame()

df["10"] = ppw10
df["20"] = ppw20
df["30"] = ppw30

print("ppw10", ppw10)
print("ppw20", ppw20)
print("ppw30", ppw30)
print("fps10", fps10)
print("fps20", fps20)
print("fps30", fps30)

print(ppw10)
print(ppw15)
print(ppw20)
print(ppw25)
print(ppw30)
print(fps10)
print(fps15)
print(fps20)
print(fps25)
print(fps30)


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

########################
import matplotlib.pyplot as plt
import numpy as np

# Data for the bar graphs
categories = ['10°C', '20°C', '30°C']  # Y-axis labels
values1 = [
    [myfpsDict["def20.0"][0].mean(), myfpsDict["def30.0"][0].mean(), myfpsDict["def40.0"][0].mean()],  # Group 1
    [myfpsDict["zTT20.0"][0].mean(), myfpsDict["zTT30.0"][0].mean(), myfpsDict["zTT40.0"][0].mean()],  # Group 1
    [myfpsDict["gea20.0"][0].mean(), myfpsDict["gea30.0"][0].mean(), myfpsDict["gea40.0"][0].mean()],  # Group 1
    [myfpsDict["DVF20.0"][0].mean(), myfpsDict["DVF30.0"][0].mean(), myfpsDict["DVF40.0"][0].mean()],  # Group 1
 ]
values2 = [
    [mydict["def20.0"][0].mean(), mydict["def30.0"][0].mean(), mydict["def40.0"][0].mean()],  # Group 1
    [mydict["zTT20.0"][0].mean(), mydict["zTT30.0"][0].mean(), mydict["zTT40.0"][0].mean()],  # Group 1
    [mydict["gea20.0"][0].mean(), mydict["gea30.0"][0].mean(), mydict["gea40.0"][0].mean()],  # Group 1
    [mydict["DVF20.0"][0].mean(), mydict["DVF30.0"][0].mean(), mydict["DVF40.0"][0].mean()],  # Group 1
]

labelLi = ["def", "zTT", "gear", "ours"]
values1 = np.array(values1).T
values2 = np.array(values2).T


# Number of groups per category
num_groups = len(values1[0])
bar_width = 0.20  # Width of individual bars
y_positions = np.arange(len(categories))  # Position for each category

# Create figure and two side-by-side subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharey=True)

# First subplot on the left
for i in range(num_groups):
    ax1.barh(y_positions + i * bar_width, [row[i] for row in values1], 
             height=bar_width, label=labelLi[i])
ax1.invert_xaxis()  # Invert to keep bars directed toward the shared y-axis
ax1.set_yticks(y_positions + bar_width)  # Set y-ticks for categories
ax1.set_yticklabels(["", "", ""])  # Empty labels for ax1
ax1.yaxis.tick_right()  # Move y-ticks to the right side of ax1
ax1.set_xlabel("FPS")
ax1.legend()


# Second subplot on the right
for i in range(num_groups):
    ax2.barh(y_positions + i * bar_width, [row[i] for row in values2], 
             height=bar_width, label=labelLi[i])
ax2.set_yticks(y_positions + bar_width)  # Align y-ticks
ax2.set_yticklabels(categories)  # Set categories on the shared middle y-axis
ax2.set_xlabel("PPW")


# Manually adjust layout for the correct appearance without tight_layout
plt.tight_layout(pad=0.5)

# Add space between y-axis labels and the bars in ax2
ax1.yaxis.set_tick_params(pad=9)  # Increase padding between labels and bars

plt.show()

#####################################
