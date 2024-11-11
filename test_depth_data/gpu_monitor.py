import subprocess
import time
import os
from tabulate import tabulate

def monitor_gpu_usage(interval=3):
    """Monitor GPU memory usage and temperature every 'interval' seconds."""
    try:
        while True:
            # Run nvidia-smi command to check memory and temperature
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,temperature.gpu', 
                                     '--format=csv,noheader,nounits'], 
                                    stdout=subprocess.PIPE)
            # Decode the result
            output = result.stdout.decode('utf-8').strip().split("\n")
            
            # Prepare table data
            table_data = []
            for line in output:
                gpu_index, memory_used, temperature = line.split(", ")
                table_data.append([gpu_index, memory_used + " MiB", temperature + " °C"])
            
            # Print table with labels
            print("\nGPU Usage and Temperature:")
            print(tabulate(table_data, headers=["GPU Index", "Memory Used", "Temperature"], tablefmt="grid"))
            
            # Wait for the specified interval
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopping GPU monitoring...")

if __name__ == "__main__":
    print("Starting GPU monitoring...")
    monitor_gpu_usage(3)  # Monitor every 3 seconds
