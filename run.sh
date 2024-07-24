# python defaultDVFS.py
# python gearDVFS.py
# python zTT.py
# python OURS4.py

# python zTT.py --total_timesteps 100 --experiment 1 --temperature 20 --initSleep 1 --loadModel zTT__zTT__1__1721364588__exp2__temp20
# python OURS4.py --total_timesteps 100 --experiment 1 --temperature 20 --initSleep 11 --loadModel DVFStrain4__OURS4__1__1721364595__exp2__temp20
# python gearDVFS.py --total_timesteps 100 --experiment 1 --temperature 20 --initSleep 10
# python defaultDVFS.py --total_timesteps 100 --experiment 2 --temperature 20 --initSleep 10

# python OURS5.py --total_timesteps 200 --experiment 1 --temperature 20 --initSleep 1 
# python OURS4.py --total_timesteps 200 --experiment 1 --temperature 20 --initSleep 1 
# python zTT.py --total_timesteps 200 --experiment 1 --temperature 20 --initSleep 1 
# python gearDVFS.py --total_timesteps 200 --experiment 1 --temperature 20 --initSleep 1
# python defaultDVFS.py --total_timesteps 200 --experiment 1 --temperature 20 --initSleep 1

# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --tau 0.0005
# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --tau 0.05
# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --learning_starts 200
# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --learning_starts 100
# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --learning_starts 1
# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --alpha 0.02
# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --alpha 0.5
# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --buffer_size 501 --learning_starts 50
# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --batch_size 32 --learning_starts 50
# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --batch_size 128 --learning_starts 50
# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --policy_frequency 1 --learning_starts 50
# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --target_network_frequency 3 --learning_starts 50
# python tempSet.py --temp 40
# python OURS6.py --total_timesteps 501 --experiment 2 --temperature 20 --initSleep 1 --policy_lr 5e-5 --learning_starts 50



python tempSet.py --temp 35
python defaultDVFS.py --total_timesteps 1001 --experiment 1 --temperature 20 --initSleep 1
python tempSet.py --temp 35
python zTT.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2
python tempSet.py --temp 35
python gearDVFS.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2
python tempSet.py --temp 35
python OURS6.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2

python tempSet.py --temp 35
python defaultDVFS.py --total_timesteps 1001 --experiment 1 --temperature 20 --initSleep 2
python tempSet.py --temp 35
python zTT.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2
python tempSet.py --temp 35
python gearDVFS.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2
python tempSet.py --temp 35
python OURS6.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2

python tempSet.py --temp 35
python defaultDVFS.py --total_timesteps 1001 --experiment 1 --temperature 20 --initSleep 2
python defaultDVFS.py --total_timesteps 1001 --experiment 1 --temperature 20 --initSleep 2
python defaultDVFS.py --total_timesteps 1001 --experiment 1 --temperature 20 --initSleep 2

python tempSet.py --temp 35
python zTT.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2
python zTT.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2
python zTT.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2

python tempSet.py --temp 35
python gearDVFS.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2
python gearDVFS.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2
python gearDVFS.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2

python tempSet.py --temp 35
python OURS6.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2
python OURS6.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2
python OURS6.py --total_timesteps 1201 --experiment 1 --temperature 20 --initSleep 2
