import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from .ellipse_regression import fit_ellipse, cart_to_pol, get_ellipse_pts, obtain_ellipses
from .edge_detection import get_edges
from .cluster_analysis import get_clusters


def obtain_params(X, clusters, labels):
    centers = np.zeros((2, len(clusters)))
    axises = np.zeros((2, len(clusters)))
    eccentricity = np.zeros(len(clusters))
    angles = np.zeros(len(clusters))
    areas = np.zeros(len(clusters))
    errors = np.zeros(len(clusters))

    for i in range(len(clusters)):
        indx = np.where(labels == clusters[i])[0]
        x = X[indx, 0]
        y = X[indx, 1]
        try:
            coeffs = fit_ellipse(x, y)
        except Exception as e:
            continue
        if len(coeffs) != 6:
            #print(count)
            continue
        x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
        centers[:, i] = x0, y0
        axises[:, i] = ap, bp
        eccentricity[i] = e
        angles[i] = phi
        areas[i] = np.pi*ap*bp
        x_fit, y_fit = get_ellipse_pts((x0, y0, ap, bp, phi), npts=len(x))
        errors[i] = np.sqrt(np.sum(y - y_fit)**2 + np.sum(x - x_fit)**2)

    return centers, axises, eccentricity, angles, areas, errors


def ellipse_image(ellipses):
    ellips_img = np.zeros((800, 1280), dtype=np.uint8)
    el_ind = ellipses.astype(int)
    for i in range(len(el_ind)):
        x, y = el_ind[i]
        y = 799 if y > 799 else y
        x = 1279 if x > 1279 else x
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y
        ellips_img[y, x] = 225
    return ellips_img


def frame_eval(path_to_video, frame, save_path):
    vidcap = cv2.VideoCapture(path_to_video)
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, img = vidcap.read()
    img = img[:,:,0]
    X = get_edges(img)
    clusters, labels = get_clusters(X)
    ellipses = obtain_ellipses(X, clusters, labels)
    ellips_img = ellipse_image(ellipses)
    dst = cv2.add(img, ellips_img)
    plt.imshow(dst, cmap='gray')
    plt.savefig(save_path, dpi=200)


def analys_video(path, frames_to_skip=30):
    centers = np.array([[], []])
    axises = np.array([[], []])
    eccentricity = np.array([])
    angles = np.array([])
    areas = np.array([])
    errors = np.array([])
    frame = np.array([])


    vidcap = cv2.VideoCapture(path)
    length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    for count in range(0, length, frames_to_skip):
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
        ret, img = vidcap.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        X = get_edges(img)
        clusters, labels = get_clusters(X)
        img_centers, img_axises, img_eccentricity, img_angles, img_areas, img_errors = obtain_params(X, clusters, labels)
        
        centers = np.append(centers, img_centers, axis=1)
        axises = np.append(axises, img_axises, axis=1)
        eccentricity = np.append(eccentricity, img_eccentricity)
        angles = np.append(angles, img_angles)
        areas = np.append(areas, img_areas)
        errors = np.append(errors, img_errors)
        frame = np.append(frame, len(clusters)*[count])

    results = pd.DataFrame({'frame': frame, 'x0': centers[0], 'y0': centers[1], 'MinorAxis':axises[1], 'MajorAxis':axises[0],
                            'eccentricity': eccentricity, 'angle': angles, 'area': areas, 'errors': errors})
    return results