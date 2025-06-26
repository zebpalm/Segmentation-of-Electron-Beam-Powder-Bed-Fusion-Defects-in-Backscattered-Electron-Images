import cv2
import numpy as np
import os
from pathlib import Path
import time
from gradient_thresholding_2 import (
    load_image, preprocess_image, compute_gradient_magnitude,
    compute_gray_gradient_distribution, find_peak_and_valley
)

def measure_segmentation_speed(image_path):
    """
    Measure the processing speed of each step in the gradient thresholding pipeline.
    Only measures time taken to create the binary mask.
    
    Args:
        image_path: Path to the input image
    
    Returns:
        Dictionary containing timing information for each step
    """
    # Initialize timing dictionary
    timings = {
        'total_time': 0,
        'read_image': 0,
        'preprocess_image': 0,
        'compute_gradient': 0,
        'compute_distribution': 0,
        'find_peak_valley': 0,
        'threshold_image': 0,
        'morphological_operations': 0,
        'remove_large_regions': 0,
        'save_mask': 0
    }
    
    start_total = time.time()
    
    # Read image
    start = time.time()
    img_original = load_image(image_path)
    timings['read_image'] = time.time() - start
    
    # Preprocess image
    start = time.time()
    img_filtered = preprocess_image(img_original)
    timings['preprocess_image'] = time.time() - start
    
    # Compute gradient magnitude
    start = time.time()
    gradient_magnitude = compute_gradient_magnitude(img_filtered)
    timings['compute_gradient'] = time.time() - start
    
    # Compute gray gradient distribution
    start = time.time()
    _, g_avg = compute_gray_gradient_distribution(img_filtered, gradient_magnitude)
    timings['compute_distribution'] = time.time() - start
    
    # Find peak G, valley Gv, peak Gp, and threshold T*
    start = time.time()
    G, Gv, Gp, T_star = find_peak_and_valley(g_avg)
    timings['find_peak_valley'] = time.time() - start
    
    # Create binary mask using T* as threshold
    start = time.time()
    _, binary_mask = cv2.threshold(img_filtered, T_star, 255, cv2.THRESH_BINARY_INV)
    timings['threshold_image'] = time.time() - start
    
    # Apply morphological opening
    start = time.time()
    kernel = np.ones((3,3), np.uint8)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    timings['morphological_operations'] = time.time() - start
    
    # Remove regions larger than 300 pixels
    start = time.time()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] > 300:
            binary_mask[labels == i] = 0
    timings['remove_large_regions'] = time.time() - start
    
    # Save binary mask
    start = time.time()
    output_dir = Path("gradient_thresholding_2/binary_masks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"filtered_{image_path.stem}.png"
    cv2.imwrite(str(output_path), binary_mask)
    timings['save_mask'] = time.time() - start
    
    # Calculate total time
    timings['total_time'] = time.time() - start_total
    
    return timings

def main():
    # Input directory
    input_dir = Path("/Users/zebpalm/Exjobb 2025/BSE images/evaluation_images/topo1_evaluation")
    
    # Process each image and collect timing information
    all_timings = []
    for img_path in input_dir.glob("*.png"):
        print(f"\nProcessing: {img_path}")
        try:
            timings = measure_segmentation_speed(img_path)
            all_timings.append(timings)
            
            # Print timing information for this image
            print("\nTiming information (milliseconds):")
            print(f"Total processing time: {timings['total_time']*1000:.1f}")
            print(f"Read image: {timings['read_image']*1000:.1f}")
            print(f"Preprocess image: {timings['preprocess_image']*1000:.1f}")
            print(f"Compute gradient: {timings['compute_gradient']*1000:.1f}")
            print(f"Compute distribution: {timings['compute_distribution']*1000:.1f}")
            print(f"Find peak/valley: {timings['find_peak_valley']*1000:.1f}")
            print(f"Threshold image: {timings['threshold_image']*1000:.1f}")
            print(f"Morphological operations: {timings['morphological_operations']*1000:.1f}")
            print(f"Remove large regions: {timings['remove_large_regions']*1000:.1f}")
            print(f"Save mask: {timings['save_mask']*1000:.1f}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Calculate and print average timings
    if all_timings:
        print("\nAverage timing information (milliseconds):")
        avg_timings = {key: sum(t[key] for t in all_timings) / len(all_timings) 
                      for key in all_timings[0].keys()}
        
        print(f"Total processing time: {avg_timings['total_time']*1000:.1f}")
        print(f"Read image: {avg_timings['read_image']*1000:.1f}")
        print(f"Preprocess image: {avg_timings['preprocess_image']*1000:.1f}")
        print(f"Compute gradient: {avg_timings['compute_gradient']*1000:.1f}")
        print(f"Compute distribution: {avg_timings['compute_distribution']*1000:.1f}")
        print(f"Find peak/valley: {avg_timings['find_peak_valley']*1000:.1f}")
        print(f"Threshold image: {avg_timings['threshold_image']*1000:.1f}")
        print(f"Morphological operations: {avg_timings['morphological_operations']*1000:.1f}")
        print(f"Remove large regions: {avg_timings['remove_large_regions']*1000:.1f}")
        print(f"Save mask: {avg_timings['save_mask']*1000:.1f}")

if __name__ == "__main__":
    main() 