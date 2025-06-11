import subprocess
import time
import sys
import signal

def get_gpu_memory_usage(gpu_index):
    command = f"nvidia-smi --query-gpu=memory.used --format=csv,nounits --id={gpu_index}"
    result = subprocess.run(command, shell=True, text=True, capture_output=True)
    memory_used = result.stdout.split('\n')[1]
    return int(memory_used)

def write_memory_usage_to_file(max_memory_used, output_file):
    with open(output_file, 'w') as file:
        file.write(f"Maximum GPU memory usage: {max_memory_used} MiB\n")

def signal_handler(signum, frame, max_memory_used, output_file):
    print(f"Received termination signal, saving maximum GPU memory usage to {output_file}")
    write_memory_usage_to_file(max_memory_used, output_file)
    sys.exit(0)

def monitor_gpu_memory(gpu_index, output_file):
    max_memory_used = 0
    signal.signal(signal.SIGTERM, lambda signum, frame: signal_handler(signum, frame, max_memory_used, output_file))
    try:
        while True:
            current_memory_used = get_gpu_memory_usage(gpu_index)
            if current_memory_used > max_memory_used:
                max_memory_used = current_memory_used
                write_memory_usage_to_file(max_memory_used, output_file)
            time.sleep(10)
    except KeyboardInterrupt:
        print(f"Final maximum GPU memory usage: {max_memory_used} MiB")
        write_memory_usage_to_file(max_memory_used, output_file)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python gpu_memory_monitor.py [gpu_index] [output_file]")
        sys.exit(1)
    
    gpu_index = sys.argv[1]
    output_file = sys.argv[2]
    monitor_gpu_memory(gpu_index, output_file)
