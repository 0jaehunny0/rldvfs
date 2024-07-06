import subprocess
import numpy as np

from time import sleep
import random

target_fps=30
target_temp=65
beta=2

little_available_frequencies = [300000, 574000, 738000, 930000, 1098000, 1197000, 1328000, 1401000, 1598000, 1704000, 1803000]
mid_available_frequencies = [400000, 553000, 696000, 799000, 910000, 1024000, 1197000, 1328000, 1491000, 1663000, 1836000, 1999000, 2130000, 2253000]
big_available_frequencies = [500000, 851000, 984000, 1106000, 1277000, 1426000, 1582000, 1745000, 1826000, 2048000, 2188000, 2252000, 2401000, 2507000, 2630000, 2704000, 2802000]
gpu_available_frequencies = [151000, 202000, 251000, 302000, 351000, 400000, 471000, 510000, 572000, 701000, 762000, 848000]

little_len = 11
mid_len = 14
big_len = 17
gpu_len = 12

little_min_freq = 300000
mid_min_freq = 400000
big_min_freq = 500000
gpu_min_freq = 151000

little_max_freq = 1803000
mid_max_freq = 2253000
big_max_freq = 2802000 
gpu_max_freq = 848000

def get_core_util():
	msg = 'adb shell cat /proc/stat'
	result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
	result = result.stdout.decode('utf-8')
	result = result.split("\n")

	li = []
	for i in result[1:9]:
		temp = np.array(i.split()[1:], dtype=np.int32)
		li.append([temp[0:7].sum(), temp[3]])

	return np.array(li)

def cal_core_util(b, a):
	x = b - a
	return (1 - x[:, 1] / x[:, 0]) 

def get_cpu_util():
	msg = 'adb shell top -n 1'
	result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
	result = result.stdout.decode('utf-8')
	result = result.split("\n")
	cpu_util = [int(result[3].split("%idle")[0].split(" ")[-1])/800*100]
	return cpu_util

def get_gpu_util():
	msg = 'adb shell cat /sys/devices/platform/1c500000.mali/utilization'
	result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
	result = result.stdout.decode('utf-8')
	result = result.split("\n")
	gpu_util = [int(result[0])/100]
	return gpu_util

def get_frequency():
	""" little """
	msg = 'adb shell cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq'
	result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
	result = result.stdout.decode('utf-8')
	result = result.split("\n")
	little = int(result[0])
	""" mid """
	msg = 'adb shell cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq'
	result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
	result = result.stdout.decode('utf-8')
	result = result.split("\n")
	mid = int(result[0])
	""" big """
	msg = 'adb shell cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq'
	result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
	result = result.stdout.decode('utf-8')
	result = result.split("\n")
	big = int(result[0])
	""" gpu """
	msg = 'adb shell cat /sys/class/misc/mali0/device/cur_freq'
	result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
	result = result.stdout.decode('utf-8')
	result = result.split("\n")
	gpu = int(result[0])

	return little, mid, big, gpu


def get_window():
    msg = 'adb shell dumpsys SurfaceFlinger'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    result = result.split("\n")
    ans = "asdf"
    for i in range(len(result)):
        if "Current layers" in result[i]:
            print(result[i+4])
            ans = result[i+4]
            break
    ans = ans.split("[")
    ans = ans[1][:-1]

    window = ans
    return ans


def get_fps(window):
    msg = 'adb shell dumpsys SurfaceFlinger --latency ' + window
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    result = result.split("\n")

    startTime = int(result[-23].split("\t")[0])
    lastTime = int(result[-3].split("\t")[-1])  
    twentyFrameTime = (lastTime - startTime) / 1000000000
    fps = 20 / twentyFrameTime

    return fps

def get_energy():
    msg = 'adb shell cat /sys/bus/iio/devices/iio:device0/energy_value'
    msg2 = 'adb shell cat /sys/bus/iio/devices/iio:device1/energy_value'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    result2 = subprocess.run(msg2.split(), stdout=subprocess.PIPE)

    result = result.stdout.decode('utf-8')
    result = result.split("\n")
    t1 = int(result[0][2:])
    big = int(result[4].split()[1])
    mid = int(result[5].split()[1])
    little = int(result[6].split()[1])
    
    result2 = result2.stdout.decode('utf-8')
    result2 = result2.split("\n")
    t2 = int(result2[0][2:])
    gpu = int(result2[7].split()[1])
    
    return t1, t2, little, mid, big, gpu

def get_battery_power():
    msg = 'adb shell cat /sys/class/power_supply/battery/current_now'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    current = int(result.stdout.decode('utf-8'))
    msg = 'adb shell cat /sys/class/power_supply/battery/voltage_now'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    voltage = int(result.stdout.decode('utf-8'))
    current = current * 1.0 / 1e3
    voltage = voltage * 1.0 / 1e6
    return [float(abs(current)*abs(voltage))] # mW

def get_temperatures():
    msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone9/temp'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    big = int(result.stdout.decode('utf-8'))
    msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone10/temp'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    mid = int(result.stdout.decode('utf-8'))
    msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone11/temp'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    little = int(result.stdout.decode('utf-8'))
    msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone12/temp'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    gpu = int(result.stdout.decode('utf-8'))
    msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone17/temp'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    qi = int(result.stdout.decode('utf-8'))
    msg = 'adb shell cat /sys/devices/virtual/thermal/thermal_zone23/temp'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    battery = int(result.stdout.decode('utf-8'))
    

    return little/1000, mid/1000, big/1000, gpu/1000, qi/1000, battery/1000

def set_frequency(little_min, little_max, mid_min, mid_max, big_min, big_max, gpu_min, gpu_max):
    """ little """
    msg = 'adb shell "echo '+str(little_min)+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(little_max)+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    """ mid """
    msg = 'adb shell "echo '+str(mid_min)+' > /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(mid_max)+' > /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    """ big """
    msg = 'adb shell "echo '+str(big_min)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(big_max)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    """ gpu """
    msg = 'adb shell "echo '+str(gpu_min)+' > /sys/class/misc/mali0/device/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(gpu_max)+' > /sys/class/misc/mali0/device/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()

def unset_frequency():
    """ little """
    msg = 'adb shell "echo '+str(little_min_freq)+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(little_max_freq)+' > /sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    """ mid """
    msg = 'adb shell "echo '+str(mid_min_freq)+' > /sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(mid_max_freq)+' > /sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    """ big """
    msg = 'adb shell "echo '+str(big_min_freq)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(big_max_freq)+' > /sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    """ gpu """
    msg = 'adb shell "echo '+str(gpu_min_freq)+' > /sys/class/misc/mali0/device/scaling_min_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()
    msg = 'adb shell "echo '+str(gpu_max_freq)+' > /sys/class/misc/mali0/device/scaling_max_freq"'
    subprocess.Popen(msg, shell=True, stdout=subprocess.PIPE).stdout.read()

def set_root(): 
    msg = 'adb root'
    subprocess.run(msg.split(), stdout=subprocess.PIPE)

def set_brightness(level):
    msg = 'adb shell settings put system screen_brightness '+str(level)
    subprocess.run(msg.split(), stdout=subprocess.PIPE)

def turn_on_screen():
    """ check screen and unlock screen """
    msg = 'adb shell dumpsys input_method | grep mInteractive=true'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    if len(result) < 1:
        """ unlock screen """
        msg = 'adb shell input keyevent 82'
        subprocess.run(msg.split(), stdout=subprocess.PIPE)
        sleep(0.5)
    a,b = 500+random.randint(0,50), 1200+random.randint(0,100)
    c,d = 500+random.randint(0,50), 400+random.randint(0,100)
    msg = 'adb shell input touchscreen swipe '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)
    subprocess.run(msg.split(), stdout=subprocess.PIPE)
    msg = 'adb shell input touchscreen swipe '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)
    subprocess.run(msg.split(), stdout=subprocess.PIPE)

def turn_off_screen():
    """ check screen and lock screen """
    msg = 'adb shell dumpsys input_method | grep mInteractive=true'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    if len(result) > 1:
        """ lock screen """
        msg = 'adb shell input keyevent 26'
        subprocess.run(msg.split(), stdout=subprocess.PIPE)
        sleep(0.5)


def turn_off_usb_charging():
    # msg = 'adb shell dumpsys battery set usb 0'
    msg = 'adb shell dumpsys battery unplug'
    subprocess.run(msg.split(), stdout=subprocess.PIPE)

def turn_on_usb_charging():
    # msg = 'adb shell dumpsys battery set usb 1'
    msg = 'adb shell dumpsys battery reset'
    subprocess.run(msg.split(), stdout=subprocess.PIPE)