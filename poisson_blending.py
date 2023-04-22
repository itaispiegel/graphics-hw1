import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


def poisson_blend(im_src, im_tgt, im_mask, center):
    # TODO: Implement Poisson blending of the source image onto the target ROI
    
    # ROI: Region of Interest  
    roi_src, roi_mask = get_src_roi_crops(im_src, im_mask)
    x_min, x_max, y_min, y_max = get_tgt_roi_bounds(center, im_tgt.shape[:2], roi_src.shape[:2])
    
    for i in range(3): 
        roi_src[:, :, i] = np.where(roi_mask == 0, im_tgt[y_min:y_max, x_min:x_max][:, :, i], roi_src[:, :, i])          
    roi_blend = blend_src(roi_src, roi_mask)
    
    cv2.imwrite('roi_src.jpg', roi_src)
    cv2.imwrite('roi_blend.jpg', roi_blend)
    
    im_blend = im_tgt # we need to return an "im_blend" variable
    im_blend[y_min:y_max, x_min:x_max] = roi_blend
    
    cv2.imwrite('poisson_blend.jpg', im_blend)
    
    return im_blend
  
   
def get_src_roi_crops(im_src: np.ndarray, im_mask: np.ndarray) -> tuple():
    h, w = im_src.shape[:2]
    
    # Get the bbox of the mask with 1-pixel padding
    indices = np.nonzero(im_mask)
    x_min, y_min = np.min(indices, axis=1)
    x_max, y_max = np.max(indices, axis=1)
    x_min = max(0, x_min - 1)
    x_max = min(h, x_max + 2)
    y_min = max(0, y_min - 1)
    y_max = min(w, y_max + 2)
     
    # Extract the ROI from the source image and mask
    roi_src = im_src[x_min:x_max, y_min:y_max, :]
    roi_mask = im_mask[x_min:x_max, y_min:y_max]
    
    return roi_src, roi_mask


def get_tgt_roi_bounds(center: tuple(), tgt_shp: tuple(), roi_shp: tuple()) -> tuple():
    x_min = max(0, center[0] - roi_shp[1]//2)
    x_max = min(tgt_shp[1], x_min + roi_shp[1])
    y_min = max(0, center[1] - roi_shp[0]//2)
    y_max = min(tgt_shp[0], y_min + roi_shp[0])
    return x_min, x_max, y_min, y_max


def blend_src(roi_src: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    h, w = roi_src.shape[:2]
    A = create_poisson_matrix(roi_mask, h, w)
    roi_blend = np.zeros((h, w, 3))
    for i in range(3):
        roi_blend[:, :, i] = spsolve(A, roi_src[:, :, i].flatten()).reshape((h, w))
        print(roi_blend[:, :, i])
    return roi_blend.astype(np.uint8)
    
    
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


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana2.bmp', help='mask file path')
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
