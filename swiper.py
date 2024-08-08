from utils import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--temp", type=float, default = 37.5,
                    help="target init battery temperature")
args = parser.parse_args()

print(args)

set_root()

while True:

    msg = 'adb shell dumpsys input_method | grep mInteractive=true'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
    if len(result) >= 1:
        a,b = 500+random.randint(0,50), 1200+random.randint(0,100)
        c,d = 500+random.randint(0,50), 400+random.randint(0,100)
        msg = 'adb shell input touchscreen swipe '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)
        subprocess.run(msg.split(), stdout=subprocess.PIPE)
        # msg = 'adb shell input touchscreen swipe '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)
        # subprocess.run(msg.split(), stdout=subprocess.PIPE)

    # randSleep = random.randint(0, 100)
    # sleep(randSleep * 20 /100)

    sleep(5)

