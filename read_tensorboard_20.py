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
    "temp/battery",
    "cstate/big",
    "cstate/mid",
    "cstate/little",
    "cstate/gpu",
    "freq/big",
    "freq/mid",
    "freq/little",
    "freq/gpu",
    "util/big"
]

scalars2 = [
    "perf/fps",
    "perf/ppw",
    "perf/power",
    "temp/gpu",
    "temp/big",
    "temp/mid",
    "temp/little",
    "temp/battery",
    "cstate/big",
    "cstate/mid",
    "cstate/little",
    "cstate/gpu",
    "real/big",
    "real/mid",
    "real/little",
    "real/gpu",
    "util/big",
    "freq/big",
    "freq/mid",
    "freq/little",
    "freq/gpu",
]

mydict = {}
ppwDict = {}
myfpsDict = {}
tempDict = {}
tempDict2 = {}
tempDict3 = {}
powerDict = {}

bigD = {}
bigC = {}
midD = {}
litD = {}
gpuD = {}
utilD = {}

bigD2 = {}
midD2 = {}
litD2 = {}
gpuD2 = {}


for subdir, dirs, files in os.walk("selected/20"):
    a = subdir.split("/")


    if len(a) > 2:
        if len(a) > 2:
            if "def" in a[2].split("_")[2]:
                mykey = a[2].split("_")[2][:3] + a[2].split("_")[4] + a[2].split("_")[-3]
            elif "zTT" in a[2].split("_")[2]:
                mykey = a[2].split("_")[2][:3] + a[2].split("_")[6] + a[2].split("_")[-5]
            elif "gea" in a[2].split("_")[2]:
                mykey = a[2].split("_")[2][:3] + a[2].split("_")[6] + a[2].split("_")[-7]
            else:
                mykey = a[2].split("_")[2][:3] + a[2].split("_")[6] + a[2].split("_")[-3]    
                

        if mykey not in mydict:
            mydict[mykey] = []
            ppwDict[mykey] = []
            myfpsDict[mykey] = []
            tempDict[mykey] = []    
            tempDict2[mykey] = []    
            tempDict3[mykey] = []    
            powerDict[mykey] = []    
            
            bigD[mykey] = []    
            bigC[mykey] = []    
            midD[mykey] = []    
            litD[mykey] = []    
            gpuD[mykey] = []    

            bigD2[mykey] = []    
            midD2[mykey] = []    
            litD2[mykey] = []    
            gpuD2[mykey] = []    

            utilD[mykey] = []    


for subdir, dirs, files in os.walk("selected/20"):
    # print(subdir)
    for file in files:

        a = subdir.split("/")
        if len(a) > 2:
            if "def" in a[2].split("_")[2]:
                mykey = a[2].split("_")[2][:3] + a[2].split("_")[4] + a[2].split("_")[-3]
            elif "zTT" in a[2].split("_")[2]:
                mykey = a[2].split("_")[2][:3] + a[2].split("_")[6] + a[2].split("_")[-5]
            elif "gea" in a[2].split("_")[2]:
                mykey = a[2].split("_")[2][:3] + a[2].split("_")[6] + a[2].split("_")[-7]
            else:
                mykey = a[2].split("_")[2][:3] + a[2].split("_")[6] + a[2].split("_")[-3]    

        filepath = subdir + os.sep + file
        if "events" in filepath:

            if "DVF" not in mykey:

                x = parse_tensorboard(filepath, scalars)
                print(subdir, x["perf/ppw"][-50:]["value"].mean())
                # print(subdir, x["perf/fps"][-10:]["value"].mean())

                mydict[mykey].append(x["perf/ppw"][:]["value"])
                powerDict[mykey].append(x["perf/power"][:]["value"])
                myfpsDict[mykey].append(x["perf/fps"][:]["value"])
                tempDict[mykey].append(x["temp/battery"][:]["value"])

                processtemp = x["temp/big"]["value"] + x["temp/mid"]["value"] + x["temp/little"]["value"] + x["temp/gpu"]["value"]
                tempDict2[mykey].append(processtemp)



                cstate = x["cstate/big"]["value"] + x["cstate/mid"]["value"] + x["cstate/little"]["value"] + x["cstate/gpu"]["value"]
                tempDict3[mykey].append(cstate)

                bigD[mykey].append(x["freq/big"]["value"])
                bigC[mykey].append(x["cstate/big"]["value"])

                midD[mykey].append(x["freq/mid"]["value"])
                litD[mykey].append(x["freq/little"]["value"])
                gpuD[mykey].append(x["freq/gpu"]["value"])
                utilD[mykey].append(x["util/big"]["value"])

            else:

                x = parse_tensorboard(filepath, scalars2)
                print(subdir, x["perf/ppw"][-50:]["value"].mean())
                # print(subdir, x["perf/fps"][-10:]["value"].mean())

                mydict[mykey].append(x["perf/ppw"][:]["value"])
                powerDict[mykey].append(x["perf/power"][:]["value"])
                myfpsDict[mykey].append(x["perf/fps"][:]["value"])
                tempDict[mykey].append(x["temp/battery"][:]["value"])

                processtemp = x["temp/big"]["value"] + x["temp/mid"]["value"] + x["temp/little"]["value"] + x["temp/gpu"]["value"]
                tempDict2[mykey].append(processtemp)
                
                cstate = x["cstate/big"]["value"] + x["cstate/mid"]["value"] + x["cstate/little"]["value"] + x["cstate/gpu"]["value"]
                tempDict3[mykey].append(cstate)

                bigD[mykey].append(x["real/big"]["value"])
                bigC[mykey].append(x["cstate/big"]["value"])
                midD[mykey].append(x["real/mid"]["value"])
                litD[mykey].append(x["real/little"]["value"])
                gpuD[mykey].append(x["real/gpu"]["value"])
                utilD[mykey].append(x["util/big"]["value"])


                if "DVF" in mykey:
                    bigD2[mykey].append(x["freq/big"]["value"])
                    midD2[mykey].append(x["freq/mid"]["value"])
                    litD2[mykey].append(x["freq/little"]["value"])
                    gpuD2[mykey].append(x["freq/gpu"]["value"])


            # mydict[mykey] = 

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



labels = ['default', 'zTT', 'gearDVFS', 'ours']
colors = ['C0', 'C1', 'C2', 'C3']

import matplotlib.pyplot as plt
import numpy as np







# Data for the box plots

b = [mydict["def30.0exp1"][0].values, mydict["zTT30.0exp1"][0].values, mydict["gea30.0exp1"][0].values, mydict["DVF30.0exp1"][0].values]


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 4), sharey=True)

a = [myfpsDict["def30.0exp1"][0].values, myfpsDict["zTT30.0exp1"][0].values, myfpsDict["gea30.0exp1"][0].values, myfpsDict["DVF30.0exp1"][0].values]
bp = ax1.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
ax1.set_xticklabels(labels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax1.set_ylabel("fps")

a = [myfpsDict["def30.0exp2"][0].values, myfpsDict["zTT30.0exp2"][0].values, myfpsDict["gea30.0exp2"][0].values, myfpsDict["DVF30.0exp2"][0].values]
bp = ax2.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
ax2.set_xticklabels(labels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

a = [myfpsDict["def30.0exp4"][0].values, myfpsDict["zTT30.0exp4"][0].values, myfpsDict["gea30.0exp4"][0].values, myfpsDict["DVF30.0exp4"][0].values]
bp = ax3.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
ax3.set_xticklabels(labels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

a = [myfpsDict["def30.0exp5"][0].values, myfpsDict["zTT30.0exp5"][0].values, myfpsDict["gea30.0exp5"][0].values, myfpsDict["DVF30.0exp5"][0].values]
bp = ax4.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
ax4.set_xticklabels(labels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.tight_layout(pad=0.5)
plt.show()


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 4), sharey=True)

a = [mydict["def30.0exp1"][0].values, mydict["zTT30.0exp1"][0].values, mydict["gea30.0exp1"][0].values, mydict["DVF30.0exp1"][0].values]
bp = ax1.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
ax1.set_xticklabels(labels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax1.set_ylabel("fps")

a = [mydict["def30.0exp2"][0].values, mydict["zTT30.0exp2"][0].values, mydict["gea30.0exp2"][0].values, mydict["DVF30.0exp2"][0].values]
bp = ax2.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
ax2.set_xticklabels(labels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

a = [mydict["def30.0exp4"][0].values, mydict["zTT30.0exp4"][0].values, mydict["gea30.0exp4"][0].values, mydict["DVF30.0exp4"][0].values]
bp = ax3.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
ax3.set_xticklabels(labels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

a = [mydict["def30.0exp5"][0].values, mydict["zTT30.0exp5"][0].values, mydict["gea30.0exp5"][0].values, mydict["DVF30.0exp5"][0].values]
bp = ax4.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
ax4.set_xticklabels(labels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.tight_layout(pad=0.5)
plt.show()





# a = [targetDict["def30.0exp4"][0].values/1000000, transform_list(targetDict["zTT30.0exp4"][0].values)/1000000, transform_list(targetDict["gea30.0exp4"][0].values)/1000000, transform_list(targetDict["DVF30.0exp4"][0].values)/1000000]
# bp = ax3.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
# # ax3.set_xticklabels(labels)
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)

# a = [targetDict["def30.0exp5"][0].values/1000000, transform_list(targetDict["zTT30.0exp5"][0].values)/1000000, transform_list(targetDict["gea30.0exp5"][0].values)/1000000, transform_list(targetDict["DVF30.0exp5"][0].values)/1000000]
# bp = ax4.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
# # ax4.set_xticklabels(labels)
# for patch, color in zip(bp['boxes'], colors):
#     patch.set_facecolor(color)
# fig.legend(["Default", "Limit"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, fontsize=22)
# plt.tight_layout(pad=0.5)


def transform_list(input_list):
    return np.array([(input_list[i] + input_list[i+1]) / 2 for i in range(0, len(input_list) - 1, 2)])

import matplotlib.patches as mpatches

targetDict = bigD

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), sharey=True)

a = [targetDict["def30.0exp1"][0].values/1000000, transform_list(targetDict["zTT30.0exp1"][0].values)/1000000, transform_list(targetDict["gea30.0exp1"][0].values)/1000000, transform_list(targetDict["DVF30.0exp1"][0].values)/1000000]
bp = ax1.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
# ax1.set_xticklabels(labels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax1.set_ylabel("Big core frequency (Ghz)")
ax1.set_xticks([])
ax1.set_xlabel("App 1")


a = [targetDict["def30.0exp2"][0].values/1000000, transform_list(targetDict["zTT30.0exp2"][0].values)/1000000, transform_list(targetDict["gea30.0exp2"][0].values)/1000000, transform_list(targetDict["DVF30.0exp2"][0].values)/1000000]
bp = ax2.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
# ax2.set_xticklabels(labels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_xticks([])
ax2.set_xlabel("App 2")


red_patch = mpatches.Patch(color='C0', label='default')
r_patch = mpatches.Patch(color='C1', label='zTT')
fig.legend(handles=[red_patch, r_patch], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, columnspacing = 4.0)

# fig.legend(["Default", "Limit"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, fontsize=22)
plt.tight_layout(pad = 0.5, rect = (0,0,1,0.95))

plt.show()





targetDict = utilD

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(6, 4), sharey=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), sharey=True)

a = [targetDict["def30.0exp1"][0].values*100, transform_list(targetDict["zTT30.0exp1"][0].values)*100, transform_list(targetDict["gea30.0exp1"][0].values)*100, transform_list(targetDict["DVF30.0exp1"][0].values)*100]
bp = ax1.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
# ax1.set_xticklabels(labels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax1.set_ylabel("Big core utilization (%)")
ax1.set_xticks([])
ax1.set_xlabel("App 1")

a = [targetDict["def30.0exp2"][0].values*100, transform_list(targetDict["zTT30.0exp2"][0].values)*100, transform_list(targetDict["gea30.0exp2"][0].values)*100, transform_list(targetDict["DVF30.0exp2"][0].values)*100]
bp = ax2.boxplot(a, patch_artist=True, medianprops=dict(color="black"), showfliers=False)
# ax2.set_xticklabels(labels)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax2.set_xticks([])
ax2.set_xlabel("App 2")


red_patch = mpatches.Patch(color='C2', label='gear')
r_patch = mpatches.Patch(color='C3', label='ear')
fig.legend(handles=[red_patch, r_patch], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, columnspacing = 4.0)

plt.tight_layout(pad = 0.5, rect = (0,0,1,0.95))

plt.show()














def transform_list(input_list):
    return np.array([(input_list[i] + input_list[i+1]) / 2 for i in range(0, len(input_list) - 1, 2)])

import matplotlib.patches as mpatches

targetDict = bigD

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), sharey=False)

a = [targetDict["def30.0exp1"][0].values/1000000, transform_list(targetDict["zTT30.0exp1"][0].values)/1000000, transform_list(targetDict["gea30.0exp1"][0].values)/1000000, transform_list(targetDict["DVF30.0exp1"][0].values)/1000000]
bp = ax1.violinplot(a, showextrema=True, showmedians=True)
for patch, color in zip(bp['bodies'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
bp['cmaxes'].set_colors(colors)
bp['cbars'].set_colors(colors)
bp['cmins'].set_colors(colors)
bp['cmedians'].set_colors("black")

# ax1.set_xticklabels(labels)
ax1.set_ylabel("Big core frequency (Ghz)")
ax1.set_xticks([])
ax1.set_xlabel("App 1")
ax1.grid(axis='y',linestyle='--')  # Add grid to the second subplot


a = [targetDict["def30.0exp2"][0].values/1000000, transform_list(targetDict["zTT30.0exp2"][0].values)/1000000, transform_list(targetDict["gea30.0exp2"][0].values)/1000000, transform_list(targetDict["DVF30.0exp2"][0].values)/1000000]
bp = ax2.violinplot(a, showextrema=True, showmedians=True)
# ax2.set_xticklabels(labels)
ax2.set_xticks([])
ax2.set_xlabel("App 2")
for patch, color in zip(bp['bodies'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
bp['cmaxes'].set_colors(colors)
bp['cbars'].set_colors(colors)
bp['cmins'].set_colors(colors)
bp['cmedians'].set_colors("black")

ax2.grid(axis='y',linestyle='--')  # Add grid to the second subplot

red_patch = mpatches.Patch(color='C0', label='default')
r_patch = mpatches.Patch(color='C1', label='zTT')
fig.legend(handles=[red_patch, r_patch], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, columnspacing = 4.0)

# fig.legend(["Default", "Limit"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, fontsize=22)
plt.tight_layout(pad = 0.5, rect = (0,0,1,0.95))

plt.show()





targetDict = utilD

# fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(6, 4), sharey=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), sharey=False)

a = [targetDict["def30.0exp1"][0].values*100, transform_list(targetDict["zTT30.0exp1"][0].values)*100, transform_list(targetDict["gea30.0exp1"][0].values)*100, transform_list(targetDict["DVF30.0exp1"][0].values)*100]
bp = ax1.violinplot(a, showextrema=True, showmedians=True)
ax1.set_ylabel("Big core utilization (%)")
ax1.set_xticks([])
ax1.set_xlabel("App 1")
for patch, color in zip(bp['bodies'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
bp['cmaxes'].set_colors(colors)
bp['cbars'].set_colors(colors)
bp['cmins'].set_colors(colors)
bp['cmedians'].set_colors("black")
a = [targetDict["def30.0exp2"][0].values*100, transform_list(targetDict["zTT30.0exp2"][0].values)*100, transform_list(targetDict["gea30.0exp2"][0].values)*100, transform_list(targetDict["DVF30.0exp2"][0].values)*100]
bp = ax2.violinplot(a, showextrema=True, showmedians=True)
ax2.set_xticks([])
ax2.set_xlabel("App 2")
for patch, color in zip(bp['bodies'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
bp['cmaxes'].set_colors(colors)
bp['cbars'].set_colors(colors)
bp['cmins'].set_colors(colors)
bp['cmedians'].set_colors("black")
ax1.grid(axis='y',linestyle='--')  # Add grid to the second subplot
ax2.grid(axis='y',linestyle='--')  # Add grid to the second subplot

red_patch = mpatches.Patch(color='C2', label='gear')
r_patch = mpatches.Patch(color='C3', label='ear')
fig.legend(handles=[red_patch, r_patch], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.035),frameon=False, borderpad=0.1, columnspacing = 4.0)

plt.tight_layout(pad = 0.5, rect = (0,0,1,0.95))

plt.show()












def transform_list(input_list):
    return np.array([(input_list[i] + input_list[i+1]) / 2 for i in range(0, len(input_list) - 1, 2)])

import matplotlib.patches as mpatches

targetDict = bigD

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), sharey=False)

a = [targetDict["def30.0exp1"][0].values/1000000, transform_list(targetDict["DVF30.0exp1"][0].values)/1000000, transform_list(bigD2["DVF30.0exp1"][0].values)/1000000]

plt.plot(a[0])
plt.plot(a[1])
plt.plot(a[2])




values = [500000, 851000, 984000, 1106000, 1277000, 1426000, 1582000, 1745000,
          1826000, 2048000, 2188000, 2252000, 2401000, 2507000, 2630000, 2704000, 2802000][::-1]

def calculate_corresponding_value(x):
    if x < 0 or x > 16:
        raise ValueError("x must be in the range [0, 16]")
    
    i = int(x)  # Lower index
    if i == 16:  # Edge case: x == 16
        return values[-1]
    
    # Linear interpolation
    value_i = values[i]
    value_next = values[i + 1]
    fractional = x - i
    return value_i + fractional * (value_next - value_i)




def transform_list(input_list):
    return np.array([(input_list[i] + input_list[i+1]) / 2 for i in range(0, len(input_list) - 1, 2)])

import matplotlib.patches as mpatches

targetDict = bigD

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), sharey=False)

bigC["def30.0exp1"][0].values

transform_list(bigC["DVF30.0exp1"][0].values)

transform_list(bigD2["DVF30.0exp1"][0].values)


transform_list(bigD["DVF30.0exp1"][0].values).mean()





a = [bigC["def30.0exp1"][0].values/1000000, transform_list(targetDict["DVF30.0exp1"][0].values)/1000000, transform_list(bigD2["DVF30.0exp1"][0].values)/1000000]

plt.plot(a[0])
plt.plot(a[1])
plt.plot(a[2])






deflim = np.array(list(map(calculate_corresponding_value, (bigC["def30.0exp1"][0].values))))
deffreq = np.array(list( bigD["def30.0exp1"][0].values))


collour = np.array(list(map(calculate_corresponding_value, transform_list(bigC["DVF30.0exp1"][0].values))))
proactive = transform_list(bigD["DVF30.0exp1"][0].values)
ourReal = transform_list(bigD2["DVF30.0exp1"][0].values)


ourThr = np.minimum(collour, proactive)


# plt.plot(collour, label = "earDVFS throttling")
# plt.plot(proactive, label = "earDVFS proactive throttling")


plt.plot(deffreq, label = "default freq")
plt.plot(deflim, label = "default throttling")
plt.plot(ourReal, label = "earDVFS real freq")
plt.plot(ourThr, label = "earDVFS throttling")
plt.legend()





little_len = 11
mid_len = 14
big_len = 17
gpu_len = 12



temp = little_len + mid_len + big_len + gpu_len

c_state_reward = []

for x in range(0, 55):
    c_state_reward.append( (((temp - x) / temp) ** 0.5) + 1)
    

c_state_reward = np.array(c_state_reward)

f20 = 20 * c_state_reward
f30 = 30 * c_state_reward
f40 = 40 * c_state_reward

plt.plot(f20)
plt.plot(f30)
plt.plot(f40)










targetDict = mydict

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10, 4), sharey=True)

a = np.array([targetDict["def30.0exp1"][0].values.mean(), targetDict["zTT30.0exp1"][0].values.mean(), targetDict["gea30.0exp1"][0].values.mean(), targetDict["DVF30.0exp1"][0].values.mean()])
# a /= 1000
bp = ax1.bar(labels, a, color = colors)
ax1.set_ylabel("ppw")
for i in range(len(a)):
    ax1.text(i,a[i],np.round(a[i], 2))

a = np.array([targetDict["def30.0exp2"][0].values.mean(), targetDict["zTT30.0exp2"][0].values.mean(), targetDict["gea30.0exp2"][0].values.mean(), targetDict["DVF30.0exp2"][0].values.mean()])
# a /= 1000
bp = ax2.bar(labels, a, color = colors)
for i in range(len(a)):
    ax2.text(i,a[i],np.round(a[i], 2))

a = np.array([targetDict["def30.0exp4"][0].values.mean(), targetDict["zTT30.0exp4"][0].values.mean(), targetDict["gea30.0exp4"][0].values.mean(), targetDict["DVF30.0exp4"][0].values.mean()])
# a /= 1000
bp = ax3.bar(labels, a, color = colors)
for i in range(len(a)):
    ax3.text(i,a[i],np.round(a[i], 2))

a = np.array([targetDict["def30.0exp5"][0].values.mean(), targetDict["zTT30.0exp5"][0].values.mean(), targetDict["gea30.0exp5"][0].values.mean(), targetDict["DVF30.0exp5"][0].values.mean()])
# a /= 1000
bp = ax4.bar(labels, a, color = colors)
for i in range(len(a)):
    ax4.text(i,a[i],np.round(a[i], 2))

plt.tight_layout(pad=0.5)
plt.show()












a = np.array([mydict["def30.0exp1"][0].values.mean(), mydict["zTT30.0exp1"][0].values.mean(), mydict["gea30.0exp1"][0].values.mean(), mydict["DVF30.0exp1"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([mydict["def30.0exp2"][0].values.mean(), mydict["zTT30.0exp2"][0].values.mean(), mydict["gea30.0exp2"][0].values.mean(), mydict["DVF30.0exp2"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([mydict["def30.0exp4"][0].values.mean(), mydict["zTT30.0exp4"][0].values.mean(), mydict["gea30.0exp4"][0].values.mean(), mydict["DVF30.0exp4"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([mydict["def30.0exp5"][0].values.mean(), mydict["zTT30.0exp5"][0].values.mean(), mydict["gea30.0exp5"][0].values.mean(), mydict["DVF30.0exp5"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)


a = np.array([myfpsDict["def30.0exp1"][0].values.mean(), myfpsDict["zTT30.0exp1"][0].values.mean(), myfpsDict["gea30.0exp1"][0].values.mean(), myfpsDict["DVF30.0exp1"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([myfpsDict["def30.0exp2"][0].values.mean(), myfpsDict["zTT30.0exp2"][0].values.mean(), myfpsDict["gea30.0exp2"][0].values.mean(), myfpsDict["DVF30.0exp2"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([myfpsDict["def30.0exp4"][0].values.mean(), myfpsDict["zTT30.0exp4"][0].values.mean(), myfpsDict["gea30.0exp4"][0].values.mean(), myfpsDict["DVF30.0exp4"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([myfpsDict["def30.0exp5"][0].values.mean(), myfpsDict["zTT30.0exp5"][0].values.mean(), myfpsDict["gea30.0exp5"][0].values.mean(), myfpsDict["DVF30.0exp5"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)



a = np.array([tempDict["def30.0exp1"][0].values.mean(), tempDict["zTT30.0exp1"][0].values.mean(), tempDict["gea30.0exp1"][0].values.mean(), tempDict["DVF30.0exp1"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([tempDict["def30.0exp2"][0].values.mean(), tempDict["zTT30.0exp2"][0].values.mean(), tempDict["gea30.0exp2"][0].values.mean(), tempDict["DVF30.0exp2"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([tempDict["def30.0exp4"][0].values.mean(), tempDict["zTT30.0exp4"][0].values.mean(), tempDict["gea30.0exp4"][0].values.mean(), tempDict["DVF30.0exp4"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([tempDict["def30.0exp5"][0].values.mean(), tempDict["zTT30.0exp5"][0].values.mean(), tempDict["gea30.0exp5"][0].values.mean(), tempDict["DVF30.0exp5"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)



a = np.array([tempDict2["def30.0exp1"][0].values.mean(), tempDict2["zTT30.0exp1"][0].values.mean(), tempDict2["gea30.0exp1"][0].values.mean(), tempDict2["DVF30.0exp1"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([tempDict2["def30.0exp2"][0].values.mean(), tempDict2["zTT30.0exp2"][0].values.mean(), tempDict2["gea30.0exp2"][0].values.mean(), tempDict2["DVF30.0exp2"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([tempDict2["def30.0exp4"][0].values.mean(), tempDict2["zTT30.0exp4"][0].values.mean(), tempDict2["gea30.0exp4"][0].values.mean(), tempDict2["DVF30.0exp4"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([tempDict2["def30.0exp5"][0].values.mean(), tempDict2["zTT30.0exp5"][0].values.mean(), tempDict2["gea30.0exp5"][0].values.mean(), tempDict2["DVF30.0exp5"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)


a = np.array([tempDict3["def30.0exp1"][0].values.mean(), tempDict3["zTT30.0exp1"][0].values.mean(), tempDict3["gea30.0exp1"][0].values.mean(), tempDict3["DVF30.0exp1"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([tempDict3["def30.0exp2"][0].values.mean(), tempDict3["zTT30.0exp2"][0].values.mean(), tempDict3["gea30.0exp2"][0].values.mean(), tempDict3["DVF30.0exp2"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([tempDict3["def30.0exp4"][0].values.mean(), tempDict3["zTT30.0exp4"][0].values.mean(), tempDict3["gea30.0exp4"][0].values.mean(), tempDict3["DVF30.0exp4"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)
a = np.array([tempDict3["def30.0exp5"][0].values.mean(), tempDict3["zTT30.0exp5"][0].values.mean(), tempDict3["gea30.0exp5"][0].values.mean(), tempDict3["DVF30.0exp5"][0].values.mean()])
print((a[-1]/a[0] - 1)*100)


a = np.array([tempDict3["def30.0exp1"][0].values.mean(), tempDict3["zTT30.0exp1"][0].values.mean(), tempDict3["gea30.0exp1"][0].values.mean(), tempDict3["DVF30.0exp1"][0].values.mean()])
print((a[-1] -a[0]))
a = np.array([tempDict3["def30.0exp2"][0].values.mean(), tempDict3["zTT30.0exp2"][0].values.mean(), tempDict3["gea30.0exp2"][0].values.mean(), tempDict3["DVF30.0exp2"][0].values.mean()])
print((a[-1] -a[0]))
a = np.array([tempDict3["def30.0exp4"][0].values.mean(), tempDict3["zTT30.0exp4"][0].values.mean(), tempDict3["gea30.0exp4"][0].values.mean(), tempDict3["DVF30.0exp4"][0].values.mean()])
print((a[-1] -a[0]))
a = np.array([tempDict3["def30.0exp5"][0].values.mean(), tempDict3["zTT30.0exp5"][0].values.mean(), tempDict3["gea30.0exp5"][0].values.mean(), tempDict3["DVF30.0exp5"][0].values.mean()])
print((a[-1] -a[0]))



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