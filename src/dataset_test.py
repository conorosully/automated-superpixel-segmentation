import numpy as np
import os
import sys
from tqdm import tqdm

def count_files(folder_path):
    """Count the .npy files in a folder"""
    return len([name for name in os.listdir(folder_path) if name.endswith('.npy')])

def check_shapes(folder_path, expected_shape):
    """Ensure all .npy files in the folder have the correct shape"""
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            if data.shape != expected_shape:
                return False
    return True

def check_binary_last_channels(folder_path, channels):
    """Check if the last two channels of each .npy file are binary"""
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            last_two_channels = data[..., -2:]
            # replace less than 0 with 0 
            last_two_channels[last_two_channels < 0] = 0
        
            if not np.all(np.isin(last_two_channels, [0, 1])):
                print(np.unique(last_two_channels))
                return False
    return True

def main(folder_path, satellite_type):
    if satellite_type == "landsat":
        expected_file_count = 30000
        expected_shape = (256, 256, 9)
    elif satellite_type == "sentinel":
        expected_file_count = 28224
        expected_shape = (256, 256, 14)
    else:
        print("Unknown satellite type. Please specify either 'sentinel' or 'landsat'.")
        return
    
    # Test 1: Count the .npy files
    file_count = count_files(folder_path)
    if file_count != expected_file_count:
        print(f"Test 1 Failed: Expected {expected_file_count} files, found {file_count}.")
    else:
        print("Test 1 Passed.")

    # Test 2: Check shapes
    if not check_shapes(folder_path, expected_shape):
        print("Test 2 Failed: Not all files have the correct shape.")
    else:
        print("Test 2 Passed.")

    # Test 3: Check if the last two channels are binary
    if not check_binary_last_channels(folder_path, 2):
        print("Test 3 Failed: Not all last two channels are binary.")
    else:
        print("Test 3 Passed.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python dataset_test.py <folder_path> <satellite_type>")
        sys.exit(1)
    folder_path = sys.argv[1]
    satellite_type = sys.argv[2]
    main(folder_path, satellite_type)