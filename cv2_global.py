import cv2

img = cv2.imread("/Users/zebpalm/Exjobb 2025/Coding/gaussian_filtering_all_results/Sample 4/ABC-capture-20241209-194134_topo02_13_gaussian_k7_s1.5.png", cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(img, 35, 255, cv2.THRESH_BINARY)
ret, inv_thresh = cv2.threshold(img, 35, 255, cv2.THRESH_BINARY_INV)
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
adpt_mean_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
adpt_gauss_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("Thresholded Image", thresh)
cv2.imshow("Inverse Thresholded Image", inv_thresh)
cv2.imshow("Original Image", img)
# cv2.imshow("Otsu", th2)
# cv2.imshow("Adaptive Mean Image", adpt_mean_thresh)
# cv2.imshow("Adaptive Gaussian Image", adpt_gauss_thresh)
cv2.waitKey(0)

