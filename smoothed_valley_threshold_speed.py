import cv2
import numpy as np
import os
from pathlib import Path
import time
from smoothed_valley_threshold import load_image, preprocess_image, find_valleys

def measure_segmentation_speed(image_path):
    """
    Measure the processing speed of each step in the valley thresholding pipeline.
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
        'compute_histogram': 0,
        'find_valleys': 0,
        'threshold_image': 0,
        'save_mask': 0
    }
    
    start_total = time.time()
    
    # Read image
    start = time.time()
    image = load_image(image_path)
    timings['read_image'] = time.time() - start
    
    # Preprocess image
    start = time.time()
    img_processed = preprocess_image(image)
    timings['preprocess_image'] = time.time() - start
    
    # Compute histogram
    start = time.time()
    hist = cv2.calcHist([img_processed], [0], None, [256], [0, 256])
    hist = hist.flatten()
    timings['compute_histogram'] = time.time() - start
    
    # Find valleys
    start = time.time()
    smoothed_hist, valleys = find_valleys(hist)
    timings['find_valleys'] = time.time() - start
    
    if valleys:
        # Use the deepest valley as threshold
        threshold, _ = valleys[0]
        
        # Threshold image
        start = time.time()
        _, binary_mask = cv2.threshold(img_processed, threshold, 255, cv2.THRESH_BINARY_INV)
        timings['threshold_image'] = time.time() - start
        
        # Save the output mask
        start = time.time()
        output_dir = Path("smoothed_valley_threshold_resemblance_attempt_2")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_mask.png"
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
            print(f"Compute histogram: {timings['compute_histogram']*1000:.1f}")
            print(f"Find valleys: {timings['find_valleys']*1000:.1f}")
            print(f"Threshold image: {timings['threshold_image']*1000:.1f}")
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
        print(f"Compute histogram: {avg_timings['compute_histogram']*1000:.1f}")
        print(f"Find valleys: {avg_timings['find_valleys']*1000:.1f}")
        print(f"Threshold image: {avg_timings['threshold_image']*1000:.1f}")
        print(f"Save mask: {avg_timings['save_mask']*1000:.1f}")

if __name__ == "__main__":
    main() 