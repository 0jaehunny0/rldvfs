import subprocess
import sys
import time

def get_fps(_window):
    msg = 'adb shell dumpsys SurfaceFlinger --latency ' + '"' + _window + '"'
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')
    result = result.split("\n")

    startTime = int(result[-23].split("\t")[0])
    lastTime = int(result[-3].split("\t")[-1])
    twentyFrameTime = (lastTime - startTime) / 1000000000
    fps = 20 / twentyFrameTime
    return fps

def get_packet_info(proc_num: int, target: str) -> tuple[int, int]:
    msg = f"adb shell cat /proc/{proc_num}/net/dev"
    result = subprocess.run(msg.split(), stdout=subprocess.PIPE)
    result = result.stdout.decode("utf-8")
    result = result.split('\n')
    result = list(filter(lambda l: 'wlan0' in l, result))[0]
    result = result.split()
    if target == "byte":
        received_packet, transmitted_packet = int(result[1]), int(result[9])
    elif target == "packet":
        received_packet, transmitted_packet = int(result[2]), int(result[1])

    return received_packet, transmitted_packet

def monitor_packets(proc_num:str, target: str, interval:int = 0.5):
    prev_received, prev_transmitted = get_packet_info(proc_num, target)
    
    while True:
        time.sleep(interval)
        cur_received, cur_transmitted = get_packet_info(proc_num, target)

        received_diff, transmitted_diff = cur_received - prev_received, cur_transmitted - prev_transmitted
        print(f"Received: {received_diff}, transmitted: {transmitted_diff}")

        prev_received, prev_transmitted = cur_received, cur_transmitted

def main():
    proc_num = sys.argv[1]
    target = sys.argv[2]
    print(f"Process num: {proc_num}")
    monitor_packets(proc_num, target)

if __name__ == '__main__':
    main()
