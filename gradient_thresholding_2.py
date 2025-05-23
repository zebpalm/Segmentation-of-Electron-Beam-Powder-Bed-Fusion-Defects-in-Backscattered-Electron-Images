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
    
    return gradient_magnitude

def compute_gray_gradient_distribution(gray_img, gradient_img):
    """
    Computes cumulative and average gradient values for each gray level (0–255).
    
    Parameters:
        gray_img (np.ndarray): Original grayscale image (uint8)
        gradient_img (np.ndarray): Gradient image (float or uint8), same shape
    
    Returns:
        g_sum: Array of cumulative gradient values for each gray level
        g_avg: Array of average gradient values for each gray level
    """
    g_sum = np.zeros(256, dtype=np.float64)
    g_count = np.zeros(256, dtype=np.int32)

    # Flatten the images for easier indexing
    gray_flat = gray_img.flatten()
    grad_flat = gradient_img.flatten()

    for i in range(len(gray_flat)):
        j = gray_flat[i]  # gray level (0–255)
        g_sum[j] += grad_flat[i]
        g_count[j] += 1

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        g_avg = np.where(g_count > 0, g_sum / g_count, 0)

    return g_sum, g_avg

def smooth_histogram(hist, sigma=2):
    """Apply Gaussian smoothing to histogram."""
    x = np.arange(len(hist))
    kernel = np.exp(-(x - x.mean())**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    smoothed_hist = np.convolve(hist, kernel, mode='same')
    return smoothed_hist

def find_peak_and_valley(g_avg, distance=40, prominence=10):
    """
    Find the peak value G from smoothed histogram, lowest valley Gv in interval [0,G],
    peak Gp in interval [0,Gv], and threshold T* = (Gp+Gv)/2.
    
    Parameters:
        g_avg (np.ndarray): Array of average gradient values
        distance (int): Minimum distance between peaks
        prominence (float): Minimum prominence of peaks
    
    Returns:
        G (int): Peak value index
        Gv (int): Lowest valley index in [0,G]
        Gp (int): Peak index in [0,Gv]
        T_star (float): Threshold value (Gp+Gv)/2
    """
    # Smooth the histogram
    smoothed_hist = smooth_histogram(g_avg, sigma=2.5)
    
    # Find the maximum in original histogram
    max_idx = np.argmax(g_avg)
    
    # Define window around the maximum
    window_start = max(0, max_idx - 60)
    window_end = max_idx + 1
    
    # Find the maximum in the window of smoothed histogram
    G = window_start + np.argmax(smoothed_hist[window_start:window_end])
    
    # Find valleys in the interval [0,G] of the smoothed histogram
    valleys, _ = find_peaks(-smoothed_hist[:G+1], distance=distance, prominence=prominence)
    
    if len(valleys) == 0:
        return G, 0, 0, 0
    
    # Find the lowest valley in [0,G]
    valley_values = smoothed_hist[valleys]
    Gv = valleys[np.argmin(valley_values)]
    
    # Find peaks in the interval [0,Gv] of the smoothed histogram
    peaks, _ = find_peaks(smoothed_hist[:Gv+1], distance=distance, prominence=prominence)
    
    if len(peaks) == 0:
        return G, Gv, 0, 0
    
    # Find the highest peak in [0,Gv]
    peak_values = smoothed_hist[peaks]
    Gp = peaks[np.argmax(peak_values)]
    
    # Calculate threshold T*
    T_star = (Gp + Gv) / 2
    
    return G, Gv, Gp, T_star

def plot_histogram(g_avg, G, Gv, Gp, T_star, output_path):
    """Plot the gray gradient distribution histogram with G, Gv, Gp, and T* markers."""
    plt.figure(figsize=(10, 6))
    
    # Plot original and smoothed histograms
    smoothed_hist = smooth_histogram(g_avg, sigma=2)
    plt.plot(g_avg, color='gray', alpha=0.3, label='Original')
    plt.plot(smoothed_hist, color='black', label='Smoothed')
    
    plt.title('Gray Gradient Distribution Histogram')
    plt.xlabel('Gray Level')
    plt.ylabel('Average Gradient')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for G, Gv, Gp, and T*
    plt.axvline(x=G, color='r', linestyle='--', label=f'G: {G}')
    plt.axvline(x=Gv, color='b', linestyle=':', label=f'Gv: {Gv}')
    plt.axvline(x=Gp, color='g', linestyle='-.', label=f'Gp: {Gp}')
    plt.axvline(x=T_star, color='m', linestyle='--', label=f'T*: {T_star:.1f}')
    
    # Add point markers at G, Gv, Gp, and T*
    plt.plot(G, smoothed_hist[G], 'ro', markersize=8, label=f'G value: {smoothed_hist[G]:.2f}')
    plt.plot(Gv, smoothed_hist[Gv], 'bo', markersize=8, label=f'Gv value: {smoothed_hist[Gv]:.2f}')
    plt.plot(Gp, smoothed_hist[Gp], 'go', markersize=8, label=f'Gp value: {smoothed_hist[Gp]:.2f}')
    plt.plot(T_star, smoothed_hist[int(T_star)], 'mo', markersize=8, label=f'T* value: {smoothed_hist[int(T_star)]:.2f}')
    
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def process_images():
    """Process all images in the input directory."""
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/evaluation_images/topo1_evaluation")
    output_dir = Path("gradient_thresholding_2")
    histogram_dir = output_dir / "gradient_distribution_histogram"
    binary_masks_dir = output_dir / "binary_masks"
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directories
    output_dir.mkdir(exist_ok=True)
    histogram_dir.mkdir(exist_ok=True)
    binary_masks_dir.mkdir(exist_ok=True)
    
    image_files = sorted(input_dir.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No PNG files found in {input_dir}")
    
    for img_path in image_files:
        try:
            # Load and preprocess image
            img_original = load_image(img_path)
            img_filtered = preprocess_image(img_original)
            
            # Compute gradient magnitude
            gradient_magnitude = compute_gradient_magnitude(img_filtered)
            
            # Compute gray gradient distribution
            _, g_avg = compute_gray_gradient_distribution(img_filtered, gradient_magnitude)
            
            # Find peak G, valley Gv, peak Gp, and threshold T*
            G, Gv, Gp, T_star = find_peak_and_valley(g_avg)
            
            # Create binary mask using T* as threshold
            # Values below T* become foreground (255), values above become background (0)
            _, binary_mask = cv2.threshold(img_filtered, T_star, 255, cv2.THRESH_BINARY_INV)
            
            # Apply morphological opening to remove small noise
            kernel = np.ones((3,3), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            
            # Remove regions larger than 300 pixels
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
            for i in range(1, num_labels):  # Skip background (label 0)
                if stats[i, cv2.CC_STAT_AREA] > 300:
                    binary_mask[labels == i] = 0
            
            # Save binary mask
            binary_mask_path = binary_masks_dir / f"filtered_{img_path.stem}.png"
            cv2.imwrite(str(binary_mask_path), binary_mask)
            
            # Plot and save histogram
            output_path = histogram_dir / f"{img_path.stem}_histogram.png"
            plot_histogram(g_avg, G, Gv, Gp, T_star, output_path)
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

if __name__ == "__main__":
    try:
        process_images()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise 