import cv2
import numpy as np
import glob
import matplotlib as mpl
import matplotlib.cm as mtpltcm
original = cv2.imread("shothole.jpg")
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
colormap = mpl.cm.jet
cNorm = mpl.colors.Normalize(vmin=0, vmax=255)
scalarMap = mtpltcm.ScalarMappable(norm=cNorm, cmap=colormap)
colors = scalarMap.to_rgba(gray)
h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # concat lists
channels = [0, 1]
original_base = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
hist_base = cv2.calcHist([original_base], channels, None, histSize, ranges, accumulate=False)
cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
# Sift and Flann
sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
all_images_to_compare = []
titles = []
histograms = []
for f in glob.iglob("image\*"):
    image = cv2.imread(f)
    base = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    test = cv2.calcHist([base], channels, None, histSize, ranges, accumulate=False)
    cv2.normalize(test, test, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    titles.append(f)
    all_images_to_compare.append(image)
    histograms.append(test)

for image_to_compare, title, histogram in zip(all_images_to_compare, titles, histograms):
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []
    for m, n in matches:
        if m.distance > 0.6*n.distance:
            good_points.append(m)
    number_keypoints = 0
    if len(kp_1) >= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)
    percentage_similarity = len(good_points) / number_keypoints * 100
    similarity = cv2.compareHist(hist_base, histogram, 2)
    print("Title: " + title)
    print("histogram similarity: " + str(int(similarity)) + "\n")
    