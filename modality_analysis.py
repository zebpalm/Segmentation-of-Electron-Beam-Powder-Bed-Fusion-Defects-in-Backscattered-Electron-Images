import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def load_image(image_path):
    """Load the image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    return img

def calculate_histogram(image):
    """Calculate normalized histogram of the image."""
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()  # Normalize
    return hist

def analyze_modality(hist, prominence_threshold=0.01, peak_height_threshold=0.2, valley_threshold=0.5):
    """Analyze if the histogram is multimodal and return detailed information."""
    # Find all peaks
    peaks, properties = find_peaks(hist, prominence=prominence_threshold)
    
    if len(peaks) == 0:
        return False, peaks, [], [], "No peaks found"
    
    # Get peak heights and prominences
    peak_heights = hist[peaks]
    prominences = properties['prominences']
    
    # Find the highest peak
    max_peak_height = max(peak_heights)
    
    # Determine significant peaks (height > threshold * max_height)
    significant_peaks = []
    significant_heights = []
    significant_prominences = []
    
    for i, (peak, height, prominence) in enumerate(zip(peaks, peak_heights, prominences)):
        if height > peak_height_threshold * max_peak_height:
            significant_peaks.append(peak)
            significant_heights.append(height)
            significant_prominences.append(prominence)
    
    # Sort peaks by position
    sorted_indices = np.argsort(significant_peaks)
    significant_peaks = [significant_peaks[i] for i in sorted_indices]
    significant_heights = [significant_heights[i] for i in sorted_indices]
    
    # Check valleys between significant peaks
    is_multimodal = False
    if len(significant_peaks) > 1:
        for i in range(len(significant_peaks)-1):
            # Get the valley between current peak and next peak
            valley_start = significant_peaks[i]
            valley_end = significant_peaks[i+1]
            valley_region = hist[valley_start:valley_end]
            
            # Find minimum in valley
            valley_min = np.min(valley_region)
            valley_min_pos = valley_start + np.argmin(valley_region)
            
            # Check if valley is deep enough
            peak1_height = significant_heights[i]
            peak2_height = significant_heights[i+1]
            valley_threshold_height = valley_threshold * min(peak1_height, peak2_height)
            
            if valley_min < valley_threshold_height:
                is_multimodal = True
                break
    
    modality = "Multimodal" if is_multimodal else "Unimodal"
    
    return is_multimodal, peaks, significant_peaks, significant_heights, modality

def plot_modality_analysis(hist, peaks, significant_peaks, significant_heights, 
                         output_path, title="Gray Level Distribution"):
    """Plot histogram with detailed modality analysis."""
    plt.figure(figsize=(15, 8))
    
    # Plot histogram
    plt.plot(hist, color='black', label='Histogram')
    
    # Plot all peaks
    plt.plot(peaks, hist[peaks], "x", color='gray', markersize=8, 
            label='All Peaks', alpha=0.5)
    
    # Plot significant peaks
    if significant_peaks:
        plt.plot(significant_peaks, significant_heights, "x", color='red', 
                markersize=12, label='Significant Peaks')
        
        # Plot valleys between significant peaks
        for i in range(len(significant_peaks)-1):
            valley_start = significant_peaks[i]
            valley_end = significant_peaks[i+1]
            valley_region = hist[valley_start:valley_end]
            valley_min = np.min(valley_region)
            valley_min_pos = valley_start + np.argmin(valley_region)
            
            # Plot valley minimum
            plt.plot(valley_min_pos, valley_min, "o", color='blue', 
                    markersize=8, label='Valley' if i == 0 else "")
            
            # Add valley threshold line
            peak1_height = significant_heights[i]
            peak2_height = significant_heights[i+1]
            valley_threshold = 0.5 * min(peak1_height, peak2_height)
            plt.axhline(y=valley_threshold, color='green', linestyle='--', 
                       alpha=0.3, label='Valley Threshold' if i == 0 else "")
    
    # Add peak information
    for i, (peak, height) in enumerate(zip(significant_peaks, significant_heights)):
        plt.text(peak, height, 
                f'Peak {i+1}\n({peak}, {height:.3f})',
                ha='center', va='bottom')
    
    plt.title(title)
    plt.xlabel('Gray Level')
    plt.ylabel('Normalized Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def process_images(input_dir, output_dir, num_images=20):
    """Process images and analyze their modality."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = sorted(list(Path(input_dir).glob('*.png')))
    
    # Process first num_images
    for image_file in image_files[:num_images]:
        try:
            print(f"Processing {image_file.name}...")
            
            # Load image
            img = load_image(str(image_file))
            
            # Calculate histogram
            hist = calculate_histogram(img)
            
            # Analyze modality
            is_multimodal, all_peaks, significant_peaks, significant_heights, modality = \
                analyze_modality(hist)
            
            # Plot analysis
            output_file = output_path / f"modality_{image_file.name}"
            plot_modality_analysis(hist, all_peaks, significant_peaks, significant_heights,
                                str(output_file),
                                f"Modality Analysis - {image_file.name} ({modality})")
            
            # Save detailed information
            with open(output_path / f"modality_{image_file.stem}.txt", "w") as f:
                f.write(f"Image: {image_file.name}\n")
                f.write(f"Modality: {modality}\n\n")
                f.write("All Peaks:\n")
                f.write("Gray Level | Frequency\n")
                f.write("-" * 30 + "\n")
                for peak in all_peaks:
                    f.write(f"{peak:10d} | {hist[peak]:.6f}\n")
                
                f.write("\nSignificant Peaks:\n")
                f.write("Gray Level | Frequency\n")
                f.write("-" * 30 + "\n")
                for peak, height in zip(significant_peaks, significant_heights):
                    f.write(f"{peak:10d} | {height:.6f}\n")
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
    
    print("\nAnalysis complete!")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All"
    output_dir = "modality_analysis"
    
    # Process the images
    process_images(input_dir, output_dir, num_images=20) 