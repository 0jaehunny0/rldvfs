from utils import *
from time import sleep
from torch.utils.tensorboard import SummaryWriter
import time

window = get_window()
# adb root
set_root()

set_brightness(158)

unset_frequency()

run_name = "default__" + str(int(time.time()))
writer = SummaryWriter(f"runs/{run_name}")

little = []
mid = []
big = []
gpu = []
ppwLi = []

for i in range(500):

    # energy before
    t1a, t2a, littlea, mida, biga, gpua = get_energy()

    sleep(0.5)
    
    # energy after
    t1b, t2b, littleb, midb, bigb, gpub = get_energy()

    # reward - energy
    little_p = (littleb - littlea)/(t1b-t1a)
    mid_p = (midb - mida)/(t1b-t1a)
    big_p = (bigb - biga)/(t1b-t1a)
    gpu_p = (gpub - gpua)/(t2b-t2a)

    fps = get_fps(window)

    ppw = fps/(little_p + mid_p + big_p + gpu_p)

    freqs = np.array(get_frequency())

    little.append(freqs[0])
    mid.append(freqs[1])
    big.append(freqs[2])
    gpu.append(freqs[3])
    ppwLi.append(ppw)

    if i % 10 == 0 and i != 0:
        writer.add_scalar("losses/little", np.array(little)[-10:].mean(), i)
        writer.add_scalar("losses/mid", np.array(mid)[-10:].mean(), i)
        writer.add_scalar("losses/big", np.array(big)[-10:].mean(), i)
        writer.add_scalar("losses/gpu", np.array(gpu)[-10:].mean(), i)
        writer.add_scalar("losses/ppw", np.array(ppwLi)[-10:].mean(), i)
        