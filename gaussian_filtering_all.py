import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

# Configuration parameters
CONFIG = {
    'num_images': 50,  # Number of images to process (set to None for all images)
    'kernel_size': 7,   # Gaussian kernel size
    'sigma': 1.5,       # Gaussian sigma value
    'input_dir': Path("/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All"),
    'output_dir': Path("gaussian_filtering_all_results")
}

# Define sample areas (y1, y2, x1, x2)
sample_areas = {
    'Sample 1': (130, 273, 128, 271),
    'Sample 2': (130, 273, 329, 472),
    'Sample 3': (329, 472, 128, 271),
    'Sample 4': (329, 472, 329, 472)
}

def load_image(image_path):
    """Load and preprocess the image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img

def extract_sample(image, sample_name):
    """Extract a specific sample area from the image."""
    y1, y2, x1, x2 = sample_areas[sample_name]
    return image[y1:y2, x1:x2]

def apply_gaussian_filtering(image):
    """Apply Gaussian filtering with configured parameters."""
    return cv2.GaussianBlur(image, (CONFIG['kernel_size'], CONFIG['kernel_size']), CONFIG['sigma'])

def plot_results(original, filtered, output_path):
    """Plot and save results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Original histogram
    hist, bins = np.histogram(original.flatten(), 256, [0,256])
    axes[0, 1].plot(hist, color='black')
    axes[0, 1].set_title('Original Histogram')
    axes[0, 1].set_xlim([0, 256])
    axes[0, 1].set_ylim([0, hist.max() * 1.1])
    
    # Filtered image
    axes[1, 0].imshow(filtered, cmap='gray')
    axes[1, 0].set_title(f'Gaussian Filter (k={CONFIG["kernel_size"]}, σ={CONFIG["sigma"]})')
    axes[1, 0].axis('off')
    
    # Filtered histogram
    hist, bins = np.histogram(filtered.flatten(), 256, [0,256])
    axes[1, 1].plot(hist, color='black')
    axes[1, 1].set_title(f'Filtered Histogram (k={CONFIG["kernel_size"]}, σ={CONFIG["sigma"]})')
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].set_ylim([0, hist.max() * 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_images():
    """Process images to apply Gaussian filtering."""
    # Clear and recreate output directory
    if CONFIG['output_dir'].exists():
        shutil.rmtree(CONFIG['output_dir'])
    CONFIG['output_dir'].mkdir(exist_ok=True)
    
    # Get list of image files
    image_files = sorted(CONFIG['input_dir'].glob("*.png"))
    print(f"Found {len(image_files)} .png files")
    
    # Limit number of images if specified
    if CONFIG['num_images'] is not None:
        image_files = image_files[:CONFIG['num_images']]
    print(f"Processing {len(image_files)} images")
    
    # Process images with progress bar
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            img = load_image(img_path)
            
            # Process each sample area
            for sample_name in sample_areas:
                # Extract sample
                sample = extract_sample(img, sample_name)
                
                # Apply Gaussian filtering
                filtered = apply_gaussian_filtering(sample)
                
                # Save results
                sample_output_dir = CONFIG['output_dir'] / sample_name
                sample_output_dir.mkdir(exist_ok=True)
                
                # Save visualization
                output_path = sample_output_dir / f"{img_path.stem}_gaussian_filtering.png"
                plot_results(sample, filtered, output_path)
                
                # Save the filtered image
                cv2.imwrite(
                    str(sample_output_dir / f"{img_path.stem}_gaussian_k{CONFIG['kernel_size']}_s{CONFIG['sigma']}.png"),
                    filtered
                )
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting Gaussian filtering script...")
    print(f"Configuration:")
    print(f"- Number of images to process: {CONFIG['num_images']}")
    print(f"- Gaussian parameters:")
    print(f"  - kernel_size: {CONFIG['kernel_size']}")
    print(f"  - sigma: {CONFIG['sigma']}")
    
    process_images()
    print("Script completed") 