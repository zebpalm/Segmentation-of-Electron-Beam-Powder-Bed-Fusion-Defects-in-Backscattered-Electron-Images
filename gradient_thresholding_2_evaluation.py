import cv2
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

def load_mask(mask_path):
    """Load and ensure binary mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    return (mask > 0).astype(np.uint8)  # Ensure binary

def calculate_metrics(pred_mask, truth_mask):
    """Calculate evaluation metrics."""
    # Flatten masks for metric calculation
    pred_flat = pred_mask.flatten()
    truth_flat = truth_mask.flatten()
    
    # Calculate metrics
    precision = precision_score(truth_flat, pred_flat)
    recall = recall_score(truth_flat, pred_flat)
    f1 = f1_score(truth_flat, pred_flat)
    jaccard = jaccard_score(truth_flat, pred_flat)
    
    # Calculate Dice coefficient
    dice = (2 * np.sum(pred_flat * truth_flat)) / (np.sum(pred_flat) + np.sum(truth_flat))
    
    # Calculate dissimilarity metrics
    dice_dissimilarity = 1 - dice
    jaccard_dissimilarity = 1 - jaccard
    
    return {
        'dice_dissimilarity': dice_dissimilarity,
        'jaccard_dissimilarity': jaccard_dissimilarity,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def evaluate_masks():
    """Evaluate prediction masks against ground truth masks."""
    # Define paths
    pred_dir = Path("/Users/zebpalm/Exjobb 2025/Coding/gradient_thresholding_2/binary_masks")
    truth_dir = Path("/Users/zebpalm/Exjobb 2025/Coding/binary_masks/William/pore")
    
    if not pred_dir.exists() or not truth_dir.exists():
        raise FileNotFoundError("Prediction or ground truth directory not found")
    
    # Get all prediction masks
    pred_files = sorted(pred_dir.glob("*.png"))
    if not pred_files:
        raise FileNotFoundError(f"No prediction masks found in {pred_dir}")
    
    # Initialize metrics storage
    all_metrics = []
    
    # Process each prediction mask
    for pred_path in pred_files:
        try:
            # Convert prediction filename to truth filename
            # Remove 'filtered_' prefix and add '_mask' suffix
            truth_name = pred_path.stem.replace('filtered_', '') + '_mask.png'
            truth_path = truth_dir / truth_name
            
            if not truth_path.exists():
                print(f"Warning: No ground truth mask found for {pred_path.name}")
                continue
            
            # Load masks
            pred_mask = load_mask(pred_path)
            truth_mask = load_mask(truth_path)
            
            # Ensure masks have same size
            if pred_mask.shape != truth_mask.shape:
                print(f"Warning: Size mismatch for {pred_path.name}")
                continue
            
            # Calculate metrics
            metrics = calculate_metrics(pred_mask, truth_mask)
            all_metrics.append(metrics)
            
        except Exception as e:
            print(f"Error processing {pred_path.name}: {str(e)}")
            continue
    
    # Calculate and print average metrics
    if all_metrics:
        avg_metrics = {
            'dice_dissimilarity': np.mean([m['dice_dissimilarity'] for m in all_metrics]),
            'jaccard_dissimilarity': np.mean([m['jaccard_dissimilarity'] for m in all_metrics]),
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1_score': np.mean([m['f1_score'] for m in all_metrics])
        }
        
        print("\nAverage Metrics:")
        print(f"Dice Dissimilarity: {avg_metrics['dice_dissimilarity']:.4f}")
        print(f"Jaccard Dissimilarity: {avg_metrics['jaccard_dissimilarity']:.4f}")
        print(f"Precision: {avg_metrics['precision']:.4f}")
        print(f"Recall: {avg_metrics['recall']:.4f}")
        print(f"F1 Score: {avg_metrics['f1_score']:.4f}")
    else:
        print("No valid metrics were calculated")

if __name__ == "__main__":
    try:
        evaluate_masks()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise 