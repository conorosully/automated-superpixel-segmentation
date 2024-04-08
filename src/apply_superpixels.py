
import utils
import argparse
import os
import numpy as np
from tqdm import tqdm

def apply_superpixels(input_dir, output_dir,satellite):

    # List all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    print(f"Found {len(image_files)} image files.")
    print(f"Shape of image files: {np.load(os.path.join(input_dir, image_files[0])).shape}")
    
    for image_file in tqdm(image_files):
        # Construct the full path to the image file
        image_path = os.path.join(input_dir, image_file)
        
        # Load the image file
        all_bands = np.load(image_path)
        
        # Get superpixel mask
        superpixel_mask = utils.get_mask_from_bands(all_bands, satellite=satellite,threshold=0,method='slic', n_segments=500, compactness=10, sigma=1)
        superpixel_mask =superpixel_mask.squeeze() # Adjusting the label shape if necessary
        
        # Save the superpixel mask
        output_path = os.path.join(output_dir, image_file)
        combined = np.concatenate((all_bands, superpixel_mask[:, :, np.newaxis]), axis=2)
        np.save(output_path, combined)
    print(f"New shape of image files: {np.load(output_path).shape}")


def main():
    parser = argparse.ArgumentParser(description="Apply superpixels to image files.")
    parser.add_argument("--satellite", default="sentinel", help="Satellite data to be used. Choose from 'sentinel' or 'landsat'.")
    parser.add_argument("--input_dir", help="Directory containing image files.")
    parser.add_argument("--output_dir", default=None, help="Directory where processed files will be saved.")

    
    args = parser.parse_args()

    if args.output_dir is None:
        # Output formated to images to the same directory as the input images
        args.output_dir = args.input_dir

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    apply_superpixels(args.input_dir, args.output_dir,args.satellite)

if __name__ == "__main__":
    main()