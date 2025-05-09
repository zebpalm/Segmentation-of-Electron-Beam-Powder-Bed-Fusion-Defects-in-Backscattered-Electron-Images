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

def find_histogram_peaks(hist, prominence=0.01):
    """Find peaks in the histogram with specified prominence."""
    peaks, properties = find_peaks(hist, prominence=prominence)
    return peaks, properties['prominences']

def plot_histogram_with_peaks(hist, peaks, prominences, output_path, title="Gray Level Distribution"):
    """Plot histogram with marked peaks."""
    plt.figure(figsize=(12, 6))
    
    # Plot histogram
    plt.plot(hist, color='black', label='Histogram')
    
    # Mark peaks
    plt.plot(peaks, hist[peaks], "x", color='red', markersize=10, label='Peaks')
    
    # Add peak values and prominences
    for i, (peak, prominence) in enumerate(zip(peaks, prominences)):
        plt.text(peak, hist[peak], 
                f'Peak {i+1}\n({peak}, {hist[peak]:.3f})\nProm: {prominence:.3f}',
                ha='center', va='bottom')
    
    plt.title(title)
    plt.xlabel('Gray Level')
    plt.ylabel('Normalized Frequency')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def process_images(input_dir, output_dir, num_images=20):
    """Process images and analyze their histogram peaks."""
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
            
            # Find peaks
            peaks, prominences = find_histogram_peaks(hist)
            
            # Plot histogram with peaks
            output_file = output_path / f"peaks_{image_file.name}"
            plot_histogram_with_peaks(hist, peaks, prominences, str(output_file),
                                    f"Gray Level Distribution with Peaks - {image_file.name}")
            
            # Save peak information
            with open(output_path / f"peaks_{image_file.stem}.txt", "w") as f:
                f.write(f"Image: {image_file.name}\n")
                f.write(f"Number of peaks: {len(peaks)}\n\n")
                f.write("Peak Information:\n")
                f.write("Gray Level | Frequency | Prominence\n")
                f.write("-" * 40 + "\n")
                for peak, prominence in zip(peaks, prominences):
                    f.write(f"{peak:10d} | {hist[peak]:.6f} | {prominence:.6f}\n")
            
        except Exception as e:
            print(f"Error processing {image_file.name}: {str(e)}")
    
    print("\nAnalysis complete!")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    # Define input and output directories
    input_dir = "/Users/zebpalm/Exjobb 2025/BSE images/abctoanalyze/All"
    output_dir = "peak_analysis"
    
    # Process the images
    process_images(input_dir, output_dir, num_images=20) 