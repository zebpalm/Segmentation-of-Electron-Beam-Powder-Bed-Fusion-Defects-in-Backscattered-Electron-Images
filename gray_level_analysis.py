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

def plot_histogram(hist, output_path, title="Gray Level Distribution"):
    """Plot histogram."""
    plt.figure(figsize=(15, 8))
    
    # Plot histogram
    plt.plot(hist, color='black', label='Histogram')
    
    plt.title(title)
    plt.xlabel('Gray Level')
    plt.ylabel('Normalized Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def process_images(input_dir, output_dir):
    """Process all images in the input directory."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize average histogram
    avg_hist = np.zeros(256)
    total_images = 0
    modality_counts = {"Unimodal": 0, "Multimodal": 0}
    
    # Process each image
    for image_file in Path(input_dir).glob('*.png'):
        try:
            print(f"Processing {image_file.name}...")
            
            # Load image
            img = load_image(str(image_file))
            
            # Calculate histogram
            hist = calculate_histogram(img)
            
            # Analyze modality
            is_multimodal, peaks, significant_peaks, significant_heights, modality = \
                analyze_modality(hist)
            
            # Update modality counts
            modality_counts[modality] += 1
            
            # Plot individual histogram
            individual_output = output_path / f"hist_{image_file.stem}.png"
            plot_histogram(hist, str(individual_output), 
                         f"Gray Level Distribution - {image_file.name} ({modality})")
            
            # Add to average histogram
            avg_hist += hist
            total_images += 1
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
    
    # Calculate and plot average histogram
    if total_images > 0:
        avg_hist /= total_images
        
        # Analyze average histogram modality
        is_multimodal, peaks, significant_peaks, significant_heights, modality = \
            analyze_modality(avg_hist)
        
        # Plot average histogram
        avg_output = output_path / "average_histogram.png"
        plot_histogram(avg_hist, str(avg_output), 
                      f"Average Gray Level Distribution")
        
        # Save statistics
        with open(output_path / "statistics.txt", "w") as f:
            f.write(f"Total images analyzed: {total_images}\n")
            f.write(f"Unimodal images: {modality_counts['Unimodal']}\n")
            f.write(f"Multimodal images: {modality_counts['Multimodal']}\n")
            f.write("\nAverage histogram statistics:\n")
            f.write(f"Mean gray level: {np.mean(avg_hist):.4f}\n")
            f.write(f"Standard deviation: {np.std(avg_hist):.4f}\n")
            f.write(f"Modality: {modality}\n")
            if significant_peaks:
                f.write("\nSignificant peaks in average histogram:\n")
                for i, (peak, height) in enumerate(zip(significant_peaks, significant_heights)):
                    f.write(f"Peak {i+1}: Gray level = {peak}, Height = {height:.4f}\n")
    
    print("\nAnalysis complete!")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All"
    output_dir = "gray_level_analysis"
    
    # Process the images
    process_images(input_dir, output_dir) 