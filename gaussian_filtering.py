import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Define sample areas (y1, y2, x1, x2)
sample_areas = {
    'Sample 1': (130, 273, 128, 271),
    'Sample 2': (130, 273, 329, 472),
    'Sample 3': (329, 472, 128, 271),
    'Sample 4': (329, 472, 329, 472)
}

# Define Gaussian filter parameters to test
gaussian_params = [
    {'kernel_size': 3, 'sigma': 0.5},
    {'kernel_size': 5, 'sigma': 1.0},
    {'kernel_size': 7, 'sigma': 1.5},
    {'kernel_size': 9, 'sigma': 2.0}
]

def load_image(image_path):
    """Load and preprocess the image."""
    print(f"Attempting to load image: {image_path}")
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    print(f"Successfully loaded image with shape: {img.shape}")
    return img

def extract_sample(image, sample_name):
    """Extract a specific sample area from the image."""
    y1, y2, x1, x2 = sample_areas[sample_name]
    return image[y1:y2, x1:x2]

def apply_gaussian_filtering(image, kernel_size, sigma):
    """Apply Gaussian filtering with given parameters."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def plot_results(original, filtered_images, params, output_path):
    """Plot and save all results."""
    n_rows = len(filtered_images) + 1
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
    
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
    
    # Filtered images and their histograms
    for i, (filtered, param) in enumerate(zip(filtered_images, params), 1):
        # Filtered image
        axes[i, 0].imshow(filtered, cmap='gray')
        axes[i, 0].set_title(f'Gaussian Filter (k={param["kernel_size"]}, σ={param["sigma"]})')
        axes[i, 0].axis('off')
        
        # Filtered histogram
        hist, bins = np.histogram(filtered.flatten(), 256, [0,256])
        axes[i, 1].plot(hist, color='black')
        axes[i, 1].set_title(f'Filtered Histogram (k={param["kernel_size"]}, σ={param["sigma"]})')
        axes[i, 1].set_xlim([0, 256])
        axes[i, 1].set_ylim([0, hist.max() * 1.1])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_images():
    """Process images to apply Gaussian filtering with different parameters."""
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All")
    output_dir = Path("gaussian_filtering_results")
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get list of image files
    image_files = sorted(input_dir.glob("*.png"))
    print(f"Found {len(image_files)} .png files")
    
    # Process first 5 images
    image_files = image_files[:5]
    print(f"Processing first {len(image_files)} images")
    
    for img_path in image_files:
        try:
            print(f"\nProcessing image: {img_path.name}")
            # Load image
            img = load_image(img_path)
            
            # Process each sample area
            for sample_name in sample_areas:
                print(f"Processing {sample_name}")
                # Extract sample
                sample = extract_sample(img, sample_name)
                
                # Apply Gaussian filtering with different parameters
                filtered_images = []
                for params in gaussian_params:
                    filtered = apply_gaussian_filtering(
                        sample, 
                        params['kernel_size'], 
                        params['sigma']
                    )
                    filtered_images.append(filtered)
                
                # Save results
                sample_output_dir = output_dir / sample_name
                sample_output_dir.mkdir(exist_ok=True)
                
                # Save visualization
                output_path = sample_output_dir / f"{img_path.stem}_gaussian_filtering.png"
                plot_results(sample, filtered_images, gaussian_params, output_path)
                
                # Save the filtered images as separate files
                for i, (filtered, params) in enumerate(zip(filtered_images, gaussian_params), 1):
                    cv2.imwrite(
                        str(sample_output_dir / f"{img_path.stem}_gaussian_k{params['kernel_size']}_s{params['sigma']}.png"),
                        filtered
                    )
                
                print(f"Completed {sample_name} from {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting Gaussian filtering script...")
    process_images()
    print("Script completed") 