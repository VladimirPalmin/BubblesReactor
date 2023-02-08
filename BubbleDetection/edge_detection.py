import cv2
import numpy as np
from skimage.filters import frangi


def get_edges(img):
    # Enhance image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    # Top Hat Transform
    topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    # Black Hat Transform
    blackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    img = cv2.normalize(img, None, 0, 1.0, cv2.NORM_L1, dtype=cv2.CV_32F)
    img_enhanced = img + topHat - blackHat
    img_blur = cv2.GaussianBlur(img_enhanced, (5,5), 0)
    frangi_edges = frangi(img_blur)
    frangi_edges = (frangi_edges*255/np.max(frangi_edges)).astype(np.uint8)
    edges = cv2.Canny(image=frangi_edges, threshold1=130, threshold2=cv2.THRESH_OTSU)
    y, x = np.nonzero(edges)
    X = np.array((x, y))
    X = X.T
    return X
