import numpy as np
import matplotlib.pyplot as plt
from skimage import segmentation, color, io, img_as_float
import cv2

# Read and prepare the image
img_path = "/Users/zebpalm/Exjobb 2025/Coding/gaussian_filtering_all_results/Sample 4/ABC-capture-20241209-194134_topo02_13_gaussian_k7_s1.5.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img_float = img_as_float(img)

# Try different parameters for Felzenszwalb
segments_1 = segmentation.felzenszwalb(img_float, scale=100, sigma=0.5, min_size=50)
segments_2 = segmentation.felzenszwalb(img_float, scale=200, sigma=0.7, min_size=100)
segments_3 = segmentation.felzenszwalb(img_float, scale=300, sigma=1.0, min_size=150)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle('Felzenszwalb Segmentation with Different Parameters', fontsize=16)

# Original image
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].set_axis_off()

# Different segmentation results
axes[0, 1].imshow(color.label2rgb(segments_1, img, kind='avg'))
axes[0, 1].set_title(f'Scale=100, Sigma=0.5, Min Size=50\nSegments: {len(np.unique(segments_1))}')
axes[0, 1].set_axis_off()

axes[1, 0].imshow(color.label2rgb(segments_2, img, kind='avg'))
axes[1, 0].set_title(f'Scale=200, Sigma=0.7, Min Size=100\nSegments: {len(np.unique(segments_2))}')
axes[1, 0].set_axis_off()

axes[1, 1].imshow(color.label2rgb(segments_3, img, kind='avg'))
axes[1, 1].set_title(f'Scale=300, Sigma=1.0, Min Size=150\nSegments: {len(np.unique(segments_3))}')
axes[1, 1].set_axis_off()

plt.tight_layout()
plt.show()

# Print statistics
print("\nSegmentation Statistics:")
print(f"Number of segments (Scale=100): {len(np.unique(segments_1))}")
print(f"Number of segments (Scale=200): {len(np.unique(segments_2))}")
print(f"Number of segments (Scale=300): {len(np.unique(segments_3))}")

# Parameters explanation
print("\nParameter Effects:")
print("- Scale: Higher values will merge more regions (fewer segments)")
print("- Sigma: Gaussian smoothing parameter, higher values smooth more")
print("- Min_size: Minimum component size, smaller components are merged") 