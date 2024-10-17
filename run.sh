# python DVFS/freqFixed.py --total_timesteps 4501 --experiment 1 --temperature 30 --initSleep 10 --timeOut 1800 --tempSet 40.0 --big 1426000 --gpu 400000

# python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 30 --initSleep 10 --timeOut 1800 --tempSet 40.0 --big 1426000 --gpu 400000 --mid 1491000 --little 1401000
# python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 30 --initSleep 10 --timeOut 1800 --tempSet 40.0 --big 1582000 --gpu 400000 --mid 1491000 --little 1401000
# python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 30 --initSleep 10 --timeOut 1800 --tempSet 40.0 --big 1745000 --gpu 400000 --mid 1491000 --little 1401000


python DVFS/zTT.py --total_timesteps 4501 --experiment 1 --temperature 30 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 40.0 --targetTemp 65
python DVFS/gearDVFS.py --total_timesteps 4501 --experiment 1 --temperature 30 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 40.0 --targetTemp 50 --targetUtil 0.0
python DVFS/defaultDVFS.py --total_timesteps 4501 --experiment 1 --temperature 30 --initSleep 10 --timeOut 1800 --tempSet 40.0
python DVFS/OURS4.py --total_timesteps 4501 --experiment 1 --temperature 30 --initSleep 10 --timeOut 1800 --tempSet 40.0 --learning_starts 1



# python DVFS/zTT.py --total_timesteps 4501 --experiment 1 --temperature 10 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 20.0 --targetTemp 75
# python DVFS/zTT.py --total_timesteps 4501 --experiment 1 --temperature 10 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 20.0 --targetTemp 55

# python DVFS/gearDVFS.py --total_timesteps 4501 --experiment 1 --temperature 10 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 20.0 --targetTemp 50 --targetUtil -0.1
# python DVFS/gearDVFS.py --total_timesteps 4501 --experiment 1 --temperature 10 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 20.0 --targetTemp 50 --targetUtil 0.1


exit 1

python DVFS/freqFixed.py --total_timesteps 4501 --experiment 1 --temperature 20 --initSleep 10 --timeOut 900 --tempSet 20 --big 1426000 --gpu 400000

zTT
python DVFS/freqFixed.py --total_timesteps 4501 --experiment 1 --temperature 30 --initSleep 10 --timeOut 900 --tempSet 30 --big 1426000 --gpu 400000



python DVFS/gearDVFS.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 32.5 --targetTemp 45
python DVFS/gearDVFS.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 32.5 --targetTemp 75

python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --tempSet 32.5 --big 1582000 --gpu 400000 --mid 1491000 --little 1401000
# python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --tempSet 32.5 --big 2802000 --gpu 848000 --mid 2253000 --little 1803000
# python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --tempSet 32.5 --big 2802000 --gpu 848000 --mid 2253000 --little 1803000

# python DVFS/OURS4.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 32.5 --learning_starts 10
python DVFS/OURS4.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 32.5 --learning_starts 1 --seed 7
python DVFS/OURS4.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 32.5 --learning_starts 1 --seed 2023


exit 1

python DVFS/freqFixed.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 27.5 --big 1426000 --gpu 400000
python DVFS/freqFixed.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 30 --big 1426000 --gpu 400000
python DVFS/freqFixed.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 32.5 --big 1426000 --gpu 400000
python DVFS/freqFixed.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 35 --big 1426000 --gpu 400000
python DVFS/freqFixed.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 37.5 --big 1426000 --gpu 400000

python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 32.5 --big 2802000 --gpu 848000
python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 32.5 --big 2704000 --gpu 848000
python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 32.5 --big 2630000 --gpu 848000
python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 32.5 --big 2507000 --gpu 848000
python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 32.5 --big 2252000 --gpu 848000
python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 32.5 --big 2188000 --gpu 848000
python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 32.5 --big 2048000 --gpu 848000
python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 32.5 --big 1826000 --gpu 848000
python DVFS/limitDefault.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 900 --tempSet 32.5 --big 1745000 --gpu 848000


python DVFS/defaultDVFS.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 32.5
python DVFS/OURS4.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 32.5 --learning_starts 1
python DVFS/OURS4.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 32.5 --learning_starts 1 --seed 0
python DVFS/OURS4.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 32.5 --learning_starts 1 --seed 2024
python DVFS/gearDVFS2.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 32.5 --targetTemp 60
python DVFS/gearDVFS.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 32.5 --targetTemp 60
python DVFS/zTT.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 32.5 --targetTemp 65
python DVFS/zTT2.py --total_timesteps 4501 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 32.5 --targetTemp 65
python DVFS/screenoff.py

exit 1

python DVFS/defaultDVFS.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 30
python DVFS/OURS3.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 30.0 --learning_starts 10
python DVFS/OURS2fps.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 30.0 --learning_starts 10
python DVFS/OURS2norm.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 30.0 --learning_starts 10
python DVFS/OURS4.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 30.0 --learning_starts 10
python DVFS/OURS.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --tempSet 30.0
python DVFS/zTT.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 30.0 --targetTemp 65
python DVFS/zTT2.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 30.0 --targetTemp 65
python DVFS/gearDVFS2.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 30.0 --targetTemp 60
python DVFS/gearDVFS.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1800 --latency 0 --tempSet 30.0 --targetTemp 60


exit 1


# python DVFS/defaultDVFS.py --total_timesteps 3001 --experiment 1 --temperature 20 --initSleep 10 --timeOut 120 --tempSet 31.0
# python DVFS/zTT.py --total_timesteps 3001 --experiment 1 --temperature 20 --initSleep 10 --timeOut 120 --tempSet 31.0
# python DVFS/gearDVFS.py --total_timesteps 3001 --experiment 1 --temperature 20 --initSleep 10 --timeOut 120 --tempSet 31.0
# python DVFS/OURS.py --total_timesteps 3001 --experiment 1 --temperature 20 --initSleep 10 --timeOut 120 --tempSet 34

# python DVFS/defaultDVFS.py --total_timesteps 3001 --experiment 1 --temperature 20 --initSleep 10 --timeOut 120 --tempSet 25.0

# python DVFS/zTT.py --total_timesteps 3001 --experiment 1 --temperature 20 --initSleep 10 --timeOut 1200 --latency 0 --tempSet 25.0 --targetTemp 65 

# python DVFS/gearDVFS.py --total_timesteps 3001 --experiment 1 --temperature 20 --initSleep 10 --timeOut 1200 --latency 100 --tempSet 25.0 --targetTemp 60

# python DVFS/OURS.py --total_timesteps 3001 --experiment 1 --temperature 20 --initSleep 10 --timeOut 120 --tempSet 25.0



python DVFS/defaultDVFS.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --tempSet 30

# python DVFS/zTT.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 0 --tempSet 30.0 --targetTemp 65 

# python DVFS/zTT.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 100 --tempSet 30.0 --targetTemp 65

# python DVFS/zTT.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 1000 --tempSet 30.0 --targetTemp 65

# python DVFS/zTT.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 100 --tempSet 30.0 --targetTemp 75

# python DVFS/zTT.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 100 --tempSet 30.0 --targetTemp 65

# python DVFS/zTT.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 100 --tempSet 30.0 --targetTemp 55

# python DVFS/zTT.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 100 --tempSet 30.0 --targetTemp 45

# python DVFS/gearDVFS.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 0 --tempSet 30.0 --targetTemp 65

# python DVFS/gearDVFS.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 100 --tempSet 30.0 --targetTemp 65

# python DVFS/gearDVFS.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 1000 --tempSet 30.0 --targetTemp 65

python DVFS/zTT.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 0 --tempSet 30.0 --targetTemp 75
python DVFS/gearDVFS.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 0 --tempSet 30.0 --targetTemp 75



# python DVFS/gearDVFS.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 100 --tempSet 30.0 --targetTemp 65

# python DVFS/gearDVFS.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 100 --tempSet 30.0 --targetTemp 55

# python DVFS/gearDVFS.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --latency 100 --tempSet 30.0 --targetTemp 45

python DVFS/OURS.py --total_timesteps 3001 --experiment 1 --temperature 25 --initSleep 10 --timeOut 1200 --tempSet 30.0