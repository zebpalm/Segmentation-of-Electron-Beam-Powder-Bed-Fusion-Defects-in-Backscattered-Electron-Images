import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def load_image(image_path):
    """Load the grayscale image."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img

def preprocess_image(image):
    """Apply sharpening and median filter to the image."""
    # Apply sharpening
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    img_sharpened = cv2.filter2D(image, -1, kernel)
    
    # Apply median filter
    img_filtered = cv2.medianBlur(img_sharpened, 3)
    
    return img_filtered

def smooth_histogram(hist, sigma=2):
    """Apply Gaussian smoothing to histogram."""
    x = np.arange(len(hist))
    kernel = np.exp(-(x - x.mean())**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    smoothed_hist = np.convolve(hist, kernel, mode='same')
    return smoothed_hist

def find_valleys(hist, distance=30, prominence=15):
    """Find valleys in the histogram using SciPy's find_peaks."""
    # Smooth the histogram
    smoothed_hist = smooth_histogram(hist, sigma=2)
    
    # Invert the histogram to find valleys as peaks
    inverted_hist = -smoothed_hist
    
    # Find peaks in the inverted histogram
    peaks, properties = find_peaks(inverted_hist, 
                                 distance=distance,  # Minimum distance between peaks
                                 prominence=prominence)  # Minimum prominence of peaks
    
    # Convert peaks back to valleys and get their values
    valleys = [(peak, smoothed_hist[peak]) for peak in peaks]
    
    # Sort valleys by depth (value)
    valleys.sort(key=lambda x: x[1])
    
    return smoothed_hist, valleys

def plot_histogram_with_valleys(smoothed_hist, valleys, img_processed, img_original, output_path):
    """Plot histogram with detected valleys and their corresponding binary images."""
    n_valleys = len(valleys)
    fig = plt.figure(figsize=(15, 5 * (n_valleys + 1)))
    
    # Plot histogram with valleys
    plt.subplot(n_valleys + 1, 1, 1)
    plt.plot(smoothed_hist, color='gray', label='Smoothed Histogram')
    
    # Plot valleys with different colors
    colors = plt.cm.rainbow(np.linspace(0, 1, len(valleys)))
    for (x, y), color in zip(valleys, colors):
        plt.axvline(x=x, color=color, linestyle='--', alpha=0.5)
        plt.plot(x, y, 'o', color=color, markersize=5)
    
    plt.title('Smoothed Histogram with Detected Valleys')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    
    # Add legend with valley positions
    legend_elements = [plt.Line2D([0], [0], color='gray', label='Smoothed Histogram')]
    for (x, y), color in zip(valleys, colors):
        legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='--', 
                                        label=f'Valley at {x}'))
    plt.legend(handles=legend_elements)
    
    # Create and display binary images for each valley
    for i, ((threshold, _), color) in enumerate(zip(valleys, colors)):
        plt.subplot(n_valleys + 1, 1, i + 2)
        
        # Create binary image using valley as threshold
        _, binary = cv2.threshold(img_processed, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Convert original image to RGB
        img_rgb = cv2.cvtColor(img_original, cv2.COLOR_GRAY2RGB)
        
        # Create red overlay
        overlay = img_rgb.copy()
        overlay[binary == 255] = [255, 0, 0]  # Set binary mask pixels to red
        
        # Blend original image with overlay
        alpha = 0.4  # 60% opacity
        cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0, img_rgb)
        
        # Display result
        plt.imshow(img_rgb)
        plt.title(f'Overlay (Threshold = {threshold})')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def process_images():
    """Process all images in the input directory."""
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/evaluation_images/topo1_evaluation")
    output_dir = Path("valley_detection")
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create main output directory and subdirectories
    output_dir.mkdir(exist_ok=True)
    comparison_dir = output_dir / "comparison_images"
    binary_masks_dir = output_dir / "binary_masks"
    comparison_dir.mkdir(exist_ok=True)
    binary_masks_dir.mkdir(exist_ok=True)
    
    image_files = sorted(input_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")
    
    for img_path in image_files:
        try:
            # Load and preprocess image
            img_original = load_image(img_path)
            img_processed = preprocess_image(img_original)
            
            # Compute histogram
            hist = cv2.calcHist([img_processed], [0], None, [256], [0, 256])
            hist = hist.flatten()
            
            # Find valleys
            smoothed_hist, valleys = find_valleys(hist)
            
            if not valleys:
                print(f"No valleys found in {img_path}")
                continue
            
            # Use the deepest valley as threshold
            threshold, _ = valleys[0]
            
            # Create binary mask
            _, binary_mask = cv2.threshold(img_processed, threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Remove regions larger than 300 pixels
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            
            # Create a new mask with only regions smaller than 300 pixels
            filtered_mask = np.zeros_like(binary_mask)
            for i in range(1, num_labels):  # Skip background (label 0)
                if stats[i, cv2.CC_STAT_AREA] <= 300:
                    filtered_mask[labels == i] = 255
            
            # Save binary mask
            binary_mask_path = binary_masks_dir / f"filtered_{img_path.stem}.png"
            cv2.imwrite(str(binary_mask_path), filtered_mask)
            
            # Save comparison visualization
            comparison_path = comparison_dir / f"{img_path.stem}_valleys.png"
            plot_histogram_with_valleys(smoothed_hist, valleys, img_processed, img_original, comparison_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        process_images()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise 