import utils
import argparse
import os
import numpy as np
from tqdm import tqdm

def apply_superpixels(input_dir, 
                      output_dir, 
                      satellite, 
                      rgb_bands, 
                      index_name, 
                      threshold, 
                      method, 
                      min_size,
                      n_segments):
    """Apply superpixels to image files in the input directory and save the results in the output directory."""
    # List all image files
    image_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]
    print(f"Found {len(image_files)} image files.")
    print(f"Shape of first image file: {np.load(os.path.join(input_dir, image_files[0])).shape}")
    
    for image_file in tqdm(image_files):
        # Construct the full path to the image file
        image_path = os.path.join(input_dir, image_file)
        
        # Load the image file
        all_bands = np.load(image_path)
        if satellite == 'sentinel':
            spectral_bands = all_bands[:, :, :12].copy()
        elif satellite == 'landsat':
            spectral_bands = all_bands[:, :, :7].copy()

        if method == 'slic':
            kwargs = {'n_segments': n_segments}
        elif method == 'felzenszwalb':
            kwargs = {'min_size': min_size}
        else:
            kwargs = {}

        superpixel_mask = utils.get_mask_from_bands(
            spectral_bands, 
            satellite=satellite,
            rgb_bands=rgb_bands,
            index_name=index_name,
            threshold=threshold, 
            method=method,
            **kwargs)
          
        superpixel_mask = superpixel_mask.squeeze()  # Adjusting the label shape if necessary
        
        # Save the superpixel mask
        output_path = os.path.join(output_dir, image_file)
        combined = np.concatenate((all_bands, superpixel_mask[:, :, np.newaxis]), axis=2)
        
        np.save(output_path, combined)
    print(f"New band position: {combined.shape[2] - 1}")
    print(f"New shape of last saved image: {np.load(output_path).shape}")

def main():
    parser = argparse.ArgumentParser(description="Apply superpixels to image files.")
    parser.add_argument("--satellite", required=True, choices=["sentinel", "landsat"], help="Satellite data type.")
    parser.add_argument("--input_dir", required=True, help="Directory containing image files.")
    parser.add_argument("--output_dir", required=False, help="Directory where processed files will be saved.")
    parser.add_argument("--rgb_bands", nargs=3, default=["nir", "green", "blue"], help="RGB bands to use for visualization.")
    parser.add_argument("--index_name", default="NDWI", help="Index name for mask generation.")
    parser.add_argument("--threshold", type=float, default=0, help="Threshold for mask segmentation. -1 for Otsu's threshold.")
    parser.add_argument("--method", default='slic', choices=["none","slic", "felzenszwalb"], help="Superpixel segmentation method.")
    parser.add_argument("--min_size", type=int, default=60, help="Minimum size of superpixels for segmentation.")
    parser.add_argument("--n_segments", type=int, default=100, help="Number of segments for SLIC method.")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    # Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nApplying superpixels to image files with the following arguments:")
    print(vars(args))  # Print all arguments
    
    apply_superpixels(
        args.input_dir, 
        args.output_dir,
        args.satellite,
        args.rgb_bands,
        args.index_name,
        args.threshold,
        args.method,
        args.min_size,
        args.n_segments
    )

if __name__ == "__main__":
    main()