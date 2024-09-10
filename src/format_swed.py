import numpy as np
import os
import argparse
import rasterio
from tqdm import tqdm


# Process image and label files for SWED dataset

def process_npy_files(input_dir, output_dir):
    """
    Process image and label files, combining them into single files with an additional channel for the label.

    Parameters:
    - image_dir: Directory containing image files.
    - label_dir: Directory containing label files.
    - output_dir: Directory where combined files will be saved.
    """
    
    # List all image files
    image_dir = os.path.join(input_dir, 'images')
    label_dir = os.path.join(input_dir, 'labels')
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.npy')]
    print(f"Found {len(image_files)} image files.")
    
    # add loading bar
    for image_file in tqdm(image_files):
        # Construct the corresponding label file name based on the image file name
        # This assumes a naming convention where you can derive the label file name from the image file name
        label_file = image_file.replace('image', 'chip')
        
        # Construct full paths to the files
        image_path = os.path.join(image_dir, image_file)
        label_path = os.path.join(label_dir, label_file)
        output_path = os.path.join(output_dir, image_file) #use the same name as the image file
        
        # Load the image and label files
        image = np.load(image_path)
        label = np.load(label_path).squeeze() # Adjusting the label shape if necessary
        
        # Verify the shapes and combine
        if image.shape == (256, 256, 12) and label.shape == (256, 256):
            combined = np.concatenate((image, label[:, :, np.newaxis]), axis=2)
            np.save(output_path, combined)
        else:
            print(f"Error processing {image_file}: Unexpected shape.")
   

def process_tif_files(input_dir, output_dir):
    """
    Process .tif image files and their corresponding .tif label files for the test set, 
    combining them into a single array with the label as an additional channel,
    and saving the result as .npy files.
    """

    # Assume image files and label files share the same base name but are located in different directories
    image_dir = os.path.join(input_dir, 'images')
    label_dir = os.path.join(input_dir, 'labels')
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
    print(f"Found {len(image_files)} image files.")
    
    for image_file in tqdm(image_files):
        image_path = os.path.join(image_dir, image_file)
        label_file = image_file.replace('image', 'label')
        label_path = os.path.join(label_dir, label_file)
        
        if not os.path.exists(label_path):
            print(f"Label file not found for {image_file}, skipping.")
            continue
        
        # Load the image data from .tif
        with rasterio.open(image_path) as src:
            image_data = src.read()  # Rasterio reads in (channels, height, width) format
            
        # Load the label data from .tif
        with rasterio.open(label_path) as src:
            label_data = src.read(1)  # Assuming label is a single channel, read(1) reads the first band
            label_data = np.expand_dims(label_data, axis=0)  # Add a channel dimension to the label
            
        # Combine the image and label data
        combined_data = np.concatenate((image_data, label_data), axis=0)
        combined_data = np.transpose(combined_data, (1, 2, 0))  # Change to (height, width, channels) format
        
        # Save the combined data as a .npy file
        output_path = os.path.join(output_dir, image_file.replace('.tif', '.npy'))
        np.save(output_path, combined_data)
        
    print(f"Processed and saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process image and label files.")
    parser.add_argument("--mode", choices=['train', 'test'], help="Specify 'train' for training set or 'test' for test set.")
    parser.add_argument("--input_dir", help="Directory containing image files.")
    parser.add_argument("--output_dir", default=None, help="Directory where processed files will be saved.")
    
    args = parser.parse_args()

    if args.output_dir is None:
        # Output formated to images to the same directory as the input images
        args.output_dir = args.input_dir

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if args.mode == 'train':
        process_npy_files(args.input_dir, args.output_dir)
    else:  # args.mode == 'test'
        process_tif_files(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()
