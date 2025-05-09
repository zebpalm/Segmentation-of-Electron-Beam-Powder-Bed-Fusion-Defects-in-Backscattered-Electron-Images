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

# Define Sobel thresholds to test
sobel_thresholds = [0.1, 0.2, 0.3, 0.4]

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

def apply_sobel_edge_detection(image, threshold):
    """Apply Sobel edge detection with given threshold."""
    # Calculate gradients
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient magnitude to [0, 1]
    grad_mag = grad_mag / grad_mag.max()
    
    # Apply threshold
    edges = (grad_mag > threshold).astype(np.uint8) * 255
    
    return edges

def plot_results(original, gaussian_results, sobel_results, output_path):
    """Plot and save all results."""
    n_gaussian = len(gaussian_params)
    n_thresholds = len(sobel_thresholds)
    
    fig, axes = plt.subplots(n_gaussian + 1, n_thresholds + 1, figsize=(15, 15))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Column headers (thresholds)
    for j, threshold in enumerate(sobel_thresholds, 1):
        axes[0, j].text(0.5, 0.5, f'Threshold\n{threshold}', 
                       ha='center', va='center', fontsize=12)
        axes[0, j].axis('off')
    
    # Gaussian filtered images and their Sobel results
    for i, (gaussian_img, gaussian_param) in enumerate(zip(gaussian_results, gaussian_params), 1):
        # Gaussian filtered image
        axes[i, 0].imshow(gaussian_img, cmap='gray')
        axes[i, 0].set_title(f'Gaussian\nk={gaussian_param["kernel_size"]}\nσ={gaussian_param["sigma"]}')
        axes[i, 0].axis('off')
        
        # Sobel results for each threshold
        for j, threshold in enumerate(sobel_thresholds, 1):
            axes[i, j].imshow(sobel_results[i-1][j-1], cmap='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_images():
    """Process images to apply Gaussian filtering and Sobel edge detection."""
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All")
    output_dir = Path("gaussian_sobel_results")
    
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
                gaussian_results = []
                for params in gaussian_params:
                    filtered = apply_gaussian_filtering(
                        sample, 
                        params['kernel_size'], 
                        params['sigma']
                    )
                    gaussian_results.append(filtered)
                
                # Apply Sobel edge detection with different thresholds
                sobel_results = []
                for gaussian_img in gaussian_results:
                    threshold_results = []
                    for threshold in sobel_thresholds:
                        edges = apply_sobel_edge_detection(gaussian_img, threshold)
                        threshold_results.append(edges)
                    sobel_results.append(threshold_results)
                
                # Save results
                sample_output_dir = output_dir / sample_name
                sample_output_dir.mkdir(exist_ok=True)
                
                # Save visualization
                output_path = sample_output_dir / f"{img_path.stem}_gaussian_sobel.png"
                plot_results(sample, gaussian_results, sobel_results, output_path)
                
                # Save the filtered and edge detected images as separate files
                for i, (gaussian_img, gaussian_param) in enumerate(zip(gaussian_results, gaussian_params)):
                    cv2.imwrite(
                        str(sample_output_dir / f"{img_path.stem}_gaussian_k{gaussian_param['kernel_size']}_s{gaussian_param['sigma']}.png"),
                        gaussian_img
                    )
                    for j, threshold in enumerate(sobel_thresholds):
                        cv2.imwrite(
                            str(sample_output_dir / f"{img_path.stem}_gaussian_k{gaussian_param['kernel_size']}_s{gaussian_param['sigma']}_sobel_t{threshold}.png"),
                            sobel_results[i][j]
                        )
                
                print(f"Completed {sample_name} from {img_path.name}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting Gaussian-Sobel analysis script...")
    process_images()
    print("Script completed") 