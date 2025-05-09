import cv2
import numpy as np
from pathlib import Path
import shutil
import os

def create_directory_structure():
    """Create the necessary directory structure for ConSinGAN training."""
    base_dir = Path("consingan_training")
    
    # Create main directories
    dirs = [
        base_dir,
        base_dir / "TrainingSets",
        base_dir / "TrainingMasks",
        base_dir / "Output"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return base_dir

def preprocess_image(image_path, output_path, size=(256, 256)):
    """Preprocess image for ConSinGAN training."""
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    
    # Normalize to [0, 255]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Save
    cv2.imwrite(str(output_path), img)
    print(f"Saved preprocessed image to: {output_path}")

def find_images():
    """Find the original image and mask files."""
    workspace = Path("/Users/zebpalm/Exjobb 2025/Coding")
    
    # Search patterns
    patterns = {
        'original': "**/ABC-capture-20241209-194442_topo02_13.png",
        'swelling': "**/task-8-annotation-2-by-1-tag-swelling-0.png",
        'pore': "**/task-8-annotation-2-by-1-tag-pore-1.png"
    }
    
    found_files = {}
    for name, pattern in patterns.items():
        matches = list(workspace.glob(pattern))
        if matches:
            found_files[name] = matches[0]
        else:
            raise FileNotFoundError(f"Could not find {name} image with pattern: {pattern}")
    
    return found_files

def setup_training():
    """Set up the ConSinGAN training environment."""
    print("Setting up ConSinGAN training environment...")
    
    # Create directory structure
    base_dir = create_directory_structure()
    print(f"Created directory structure in: {base_dir}")
    
    # Find image files
    try:
        files = find_images()
        print("\nFound required files:")
        for name, path in files.items():
            print(f"{name}: {path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Process and save training image
    try:
        print("\nProcessing images...")
        preprocess_image(
            files['original'],
            base_dir / "TrainingSets" / "training_image.png"
        )
        
        # Process and save masks
        preprocess_image(
            files['swelling'],
            base_dir / "TrainingMasks" / "swelling_mask.png"
        )
        preprocess_image(
            files['pore'],
            base_dir / "TrainingMasks" / "pore_mask.png"
        )
    except Exception as e:
        print(f"Error processing images: {e}")
        return
    
    # Create configuration file
    config = {
        "train_mode": "generation",
        "input_name": "training_image.png",
        "mask_name": ["swelling_mask.png", "pore_mask.png"],
        "min_size": 25,
        "max_size": 256,
        "noise_amp": 0.1,
        "nc_im": 1,  # 1 for grayscale
        "scales": [1.0, 0.75, 0.5, 0.25],
        "alpha": 10,
        "beta": 0.1,
        "lambda_grad": 0.1,
        "niter": 2000,
        "batch_size": 1
    }
    
    # Save configuration
    config_path = base_dir / "config.txt"
    with open(config_path, "w") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    print(f"\nSaved configuration to: {config_path}")
    
    print("\nConSinGAN training environment set up successfully!")
    print("\nNext steps:")
    print("1. Install ConSinGAN requirements:")
    print("   pip install torch torchvision")
    print("2. Clone ConSinGAN repository:")
    print("   git clone https://github.com/tohinz/ConSinGAN.git")
    print("3. Copy the training data to the ConSinGAN directory:")
    print(f"   cp -r {base_dir}/* path/to/ConSinGAN/")
    print("4. Start training:")
    print("   cd path/to/ConSinGAN")
    print("   python main.py --input_name training_image.png")

if __name__ == "__main__":
    setup_training() 