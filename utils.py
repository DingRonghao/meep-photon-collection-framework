import os
import datetime

def make_output_dir(prefix="sim"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output/{prefix}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
