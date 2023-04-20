import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


def poisson_blend(im_src, im_tgt, im_mask, center):
    # TODO: Implement Poisson blending of the source image onto the target ROI
    # calculate Laplacian
    laplacian = calc_image_laplacian(im_src)
    
    # solve the poisson equation
    pois_solution = calc_poisson_equation(im_mask, laplacian)
    
    # blend the images
    im_blend = blend_images(im_src, im_tgt, im_mask, pois_solution, center)
    return im_blend
     
                  
def calc_image_laplacian(image: np.ndarray) -> np.ndarray:             
    laplacian = np.zeros_like(image)

    # calculate Laplacian for non-edge pixels
    laplacian[1:-1, 1:-1] = image[2:, 1:-1] + image[:-2, 1:-1] + image[1:-1, 2:] + image[1:-1, :-2] - 4 * image[1:-1, 1:-1]

    # calculate Laplacian for edge pixels
    laplacian[0, 1:-1] = image[1, 1:-1] + image[0, 2:] + image[0, :-2] - 3 * image[0, 1:-1]
    laplacian[-1, 1:-1] = image[-2, 1:-1] + image[-1, 2:] + image[-1, :-2] - 3 * image[-1, 1:-1]
    laplacian[1:-1, 0] = image[2:, 0] + image[:-2, 0] + image[1:-1, 1] - 3 * image[1:-1, 0]
    laplacian[1:-1, -1] = image[2:, -1] + image[:-2, -1] + image[1:-1, -2] - 3 * image[1:-1, -1]
    
    # calculate Laplacian for corner pixels
    laplacian[0, 0] = image[1, 0] + image[0, 1] - 2 * image[0, 0]
    laplacian[-1, 0] = image[-2, 0] + image[-1, 1] - 2 * image[-1, 0]
    laplacian[0, -1] = image[1, -1] + image[0, -2] - 2 * image[0, -1]
    laplacian[-1, -1] = image[-2, -1] + image[-1, -2] - 2 * image[-1, -1]
    
    return laplacian


def calc_poisson_equation(mask: np.ndarray, laplacian: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    mask_flat = mask.flatten()
    A = create_poisson_matrix(mask, h, w)
    x_r = spsolve(A, laplacian[:, :, 0].flatten() * mask_flat)
    x_g = spsolve(A, laplacian[:, :, 1].flatten() * mask_flat)
    x_b = spsolve(A, laplacian[:, :, 2].flatten() * mask_flat)
    return np.dstack((x_r.reshape(mask.shape[:2]), 
                      x_g.reshape(mask.shape[:2]), 
                      x_b.reshape(mask.shape[:2])))
    
    
def create_poisson_matrix(mask: np.ndarray, h: int, w:int) -> scipy.sparse.lil_matrix:
    A = scipy.sparse.lil_matrix((h*w, h*w))
    
    for y in range(h):
        for x in range(w):
            i = y*w + x
            if mask[y, x] == 0:
                A[i, i] = 1
            else:    
                A[i, i] = -4
                if y > 0:
                    j = (y-1)*w + x
                    A[i, j] = 1
                if y < h-1:
                    j = (y+1)*w + x
                    A[i, j] = 1
                if x > 0:
                    j = y*w + (x-1)
                    A[i, j] = 1
                if x < w-1:
                    j = y*w + (x+1)
                    A[i, j] = 1
                    
    return A


def get_rhs_vector(laplacian: np.ndarray, im_mask: np.ndarray) -> np.ndarray:
    # Flatten the Laplacian and mask matrices
    laplacian_flat = laplacian.reshape(-1)
    mask_flat = im_mask.flatten()
    
    # Get the indices of non-zero elements in the mask
    indices = np.nonzero(mask_flat)
    
    # Create the right-hand side vector
    b = -laplacian_flat * mask_flat
    b[indices] = 0
    
    return b


def blend_images(im_src: np.ndarray, im_tgt: np.ndarray, im_mask: np.ndarray, pois_solution: np.ndarray, center: tuple) -> np.ndarray:
    # how evey arg was ctreated:
    # im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    # im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    # im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE) and then:
    # im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]
    # center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))
    # pos_solution is the solution of the poisson equation
    
    # we need to take the object from the source image and put it on the target image such that the object is centered on the center of the target image
    # Extract region of interest from source and target images
    h, w = im_src.shape[:2]
    y_min = max(0, center[1] - h//2)
    y_max = min(im_tgt.shape[0], y_min + h)
    x_min = max(0, center[0] - w//2)
    x_max = min(im_tgt.shape[1], x_min + w)
    roi_tgt = im_tgt[y_min:y_max, x_min:x_max]
    
    # Blend the source image onto the target ROI using the Poisson solution
    blended_roi = np.uint8(pois_solution)
    blended_roi_masked = cv2.bitwise_and(blended_roi, blended_roi, mask=im_mask)
    roi_tgt_masked = cv2.bitwise_and(roi_tgt, roi_tgt, mask=cv2.bitwise_not(im_mask))
    result_roi = cv2.add(blended_roi_masked, roi_tgt_masked)
    cv2.imwrite('0.png', roi_tgt)
    cv2.imwrite('1.png', blended_roi)
    cv2.imwrite('2.png', blended_roi_masked)  
    cv2.imwrite('3.png', roi_tgt_masked)  
    cv2.imwrite('4.png', result_roi)    
    
    # Replace the target ROI with the blended image
    im_tgt[y_min:y_max, x_min:x_max] = result_roi
    return im_tgt


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
