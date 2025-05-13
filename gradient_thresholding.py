import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_image(image_path):
    """Load the grayscale image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img

def compute_gradient_magnitude(image):
    """Compute gradient magnitude using Sobel operator in 4 directions."""
    # Sobel operators for different directions
    sobel_vertical = np.array([[-1, -2, -1],
                              [0, 0, 0],
                              [1, 2, 1]])
    
    sobel_horizontal = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])
    
    sobel_diagonal1 = np.array([[-2, -1, 0],
                               [-1, 0, 1],
                               [0, 1, 2]])
    
    sobel_diagonal2 = np.array([[0, -1, -2],
                               [1, 0, -1],
                               [2, 1, 0]])
    
    # Apply Sobel operators
    grad_vertical = cv2.filter2D(image, cv2.CV_64F, sobel_vertical)
    grad_horizontal = cv2.filter2D(image, cv2.CV_64F, sobel_horizontal)
    grad_diagonal1 = cv2.filter2D(image, cv2.CV_64F, sobel_diagonal1)
    grad_diagonal2 = cv2.filter2D(image, cv2.CV_64F, sobel_diagonal2)
    
    # Compute total gradient magnitude (sum of absolute values)
    gradient_magnitude = np.abs(grad_vertical) + np.abs(grad_horizontal) + \
                        np.abs(grad_diagonal1) + np.abs(grad_diagonal2)
    
    return np.uint8(np.clip(gradient_magnitude, 0, 255))

def compute_gradient_histogram(original_image, gradient_image):
    """Compute gray gradient distribution histogram."""
    # Initialize arrays for gradient sum and count for each gray level
    gradient_sum = np.zeros(256)
    pixel_count = np.zeros(256)
    
    # Accumulate gradients for each gray level
    for gray_level in range(256):
        # Find pixels with current gray level
        mask = (original_image == gray_level)
        if np.any(mask):
            # Sum gradients at these positions
            gradient_sum[gray_level] = np.sum(gradient_image[mask])
            pixel_count[gray_level] = np.sum(mask)
    
    # Compute average gradient for each gray level
    average_gradient = np.zeros(256)
    valid_levels = pixel_count > 0
    average_gradient[valid_levels] = gradient_sum[valid_levels] / pixel_count[valid_levels]
    
    return average_gradient

def find_optimal_threshold(gradient_histogram, dg=5):
    """Find optimal threshold based on gradient histogram morphology."""
    # Find peak in gray-level histogram
    gray_levels = np.arange(256)
    G = np.argmax(gradient_histogram)
    
    # Search for minimum gradient in [G-dg, G+dg]
    search_range = slice(max(0, G-dg), min(256, G+dg+1))
    GV = G - dg + np.argmin(gradient_histogram[search_range])
    
    # Search for maximum gradient in [0, GV]
    GP = np.argmax(gradient_histogram[:GV+1])
    
    # Optimal threshold is midpoint of transition zone
    T_star = (GP + GV) // 2
    
    return T_star, GP, GV

def plot_results(original, gradient_magnitude, thresholded, gradient_histogram, 
                T_star, GP, GV, output_path):
    """Create and save visualization of results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Gradient magnitude
    axes[0, 1].imshow(gradient_magnitude, cmap='gray')
    axes[0, 1].set_title('Gradient Magnitude')
    axes[0, 1].axis('off')
    
    # Thresholded image
    axes[0, 2].imshow(thresholded, cmap='gray')
    axes[0, 2].set_title('Thresholded Image')
    axes[0, 2].axis('off')
    
    # Create RGB version of original image
    original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
    
    # Create red mask (where thresholded is white)
    red_mask = np.zeros_like(original_rgb)
    red_mask[thresholded == 255] = [255, 0, 0]  # Red color for white pixels
    
    # Overlay red mask on original image
    overlay = original_rgb.copy()
    mask_indices = thresholded == 255
    overlay[mask_indices] = red_mask[mask_indices]
    
    # Display overlay
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Red Mask Overlay')
    axes[1, 0].axis('off')
    
    # Gradient histogram with threshold markers
    axes[1, 1].plot(gradient_histogram, color='gray')
    axes[1, 1].axvline(x=T_star, color='r', linestyle='--', label=f'Threshold: {T_star}')
    axes[1, 1].axvline(x=GP, color='g', linestyle=':', label=f'GP: {GP}')
    axes[1, 1].axvline(x=GV, color='b', linestyle=':', label=f'GV: {GV}')
    axes[1, 1].set_title('Gray Gradient Distribution Histogram')
    axes[1, 1].legend()
    
    # Hide the last subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_images():
    """Process all images in the input directory."""
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/evaluation_images/topo1")
    output_dir = Path("gradient_thresholding_resemblance_attempt")
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    output_dir.mkdir(exist_ok=True)
    
    image_files = sorted(input_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")
    
    print(f"Found {len(image_files)} PNG files to process")
    
    for img_path in image_files:
        try:
            print(f"\nProcessing {img_path.name}")
            img = load_image(img_path)
            print(f"Image loaded successfully. Shape: {img.shape}")
            
            # Compute gradient magnitude
            gradient_magnitude = compute_gradient_magnitude(img)
            print("Computed gradient magnitude")
            
            # Compute gradient histogram
            gradient_histogram = compute_gradient_histogram(img, gradient_magnitude)
            print("Computed gradient histogram")
            
            # Find optimal threshold
            T_star, GP, GV = find_optimal_threshold(gradient_histogram)
            print(f"Found optimal threshold: {T_star} (GP: {GP}, GV: {GV})")
            
            # Apply threshold
            _, thresholded = cv2.threshold(img, T_star, 255, cv2.THRESH_BINARY)
            print("Applied threshold")
            
            # Save visualization
            output_path = output_dir / f"{img_path.stem}_gradient_analysis.png"
            plot_results(img, gradient_magnitude, thresholded, gradient_histogram,
                        T_star, GP, GV, output_path)
            print(f"Saved results to {output_path}")
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        process_images()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise 