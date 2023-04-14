import argparse
import itertools

import cv2
import igraph as ig
import numpy as np
from sklearn.cluster import KMeans

GC_BGD = 0  # Hard bg pixel
GC_FGD = 1  # Hard fg pixel, will not be used
GC_PR_BGD = 2  # Soft bg pixel
GC_PR_FGD = 3  # Soft fg pixel


def require_initialization(func):
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            raise ValueError("Instance not initialized")
        return func(self, *args, **kwargs)

    return wrapper


class Component:
    def __init__(self, data_points, total_points_count, mean=None):
        self.data_points = data_points
        self.mean = mean
        if self.mean is None:
            self.mean = np.mean(data_points, axis=0)
        self.covariance_matrix = np.cov(self.data_points, rowvar=False)
        self.covariance_matrix_det = np.linalg.det(self.covariance_matrix)
        self.covariance_matrix_inverse = np.linalg.inv(self.covariance_matrix)
        self.weight = len(data_points) / total_points_count

    def pdf(self, x):
        d = self.mean.shape[0]
        norm = np.sqrt((2 * np.pi) ** d * self.covariance_matrix_det)
        exponent = (
            -0.5 * (x - self.mean).T @ self.covariance_matrix_inverse @ (x - self.mean)
        )
        return (1 / norm) * np.exp(exponent)


class GaussianMixture:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self._kmeans = KMeans(n_components, n_init="auto")
        self._initialized = False
        self.components = None
        self.data_points = None

    def init(self, X):
        if self._initialized:
            raise ValueError("Already initialized")
        self._initialized = True
        self.data_points = X
        labels = self._kmeans.fit_predict(X)

        self.components = []
        for i in range(self.n_components):
            component = Component(
                X[np.where(labels == i)], len(X), mean=self._kmeans.cluster_centers_[i]
            )
            self.components.append(component)

        return self

    def assign_point_to_component(self, point):
        scores = [component.pdf(point) for component in self.components]
        return np.argmax(scores)

    def update(self, X):
        labels = np.fromiter((self.assign_point_to_component(x) for x in X), dtype=int)
        updated_components = []
        for i in range(self.n_components):
            component = Component(X[np.where(labels == i)], len(X))
            updated_components.append(component)
        self.components = updated_components

    def calc_prob(self, X):
        prob = [c.pdf(x) for x in X for c in self.components]
        weights = [c.weight for c in self.components]
        return np.dot(weights, prob)


def partition_pixels(img, mask):
    bg_pixels = img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    fg_pixels = img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)]
    return bg_pixels, fg_pixels


def partition_pixels_indexes(img, mask):
    flattened_mask = mask.reshape(-1)
    bgd_idxs = np.where(flattened_mask == GC_BGD)[0]
    fgd_idxs = np.where(flattened_mask == GC_FGD)[0]
    pr_bgd_idxs = np.where(flattened_mask == GC_PR_BGD)[0]
    pr_fgd_idxs = np.where(flattened_mask == GC_PR_FGD)[0]
    return bgd_idxs, fgd_idxs, pr_bgd_idxs, pr_fgd_idxs


# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Initalize the inner square to Foreground
    mask[y : y + h, x : x + w] = GC_PR_FGD
    mask[rect[1] + rect[3] // 2, rect[0] + rect[2] // 2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000
    for i in range(num_iters):
        # Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def initalize_GMMs(img, mask, n_components: int = 5):
    # TODO: implement initalize_GMMs
    bg_pixels, fg_pixels = partition_pixels(img, mask)
    bgGMM = GaussianMixture(n_components).init(bg_pixels)
    fgGMM = GaussianMixture(n_components).init(fg_pixels)
    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    bg_pixels, fg_pixels = partition_pixels(img, mask)
    bgGMM.update(bg_pixels)
    fgGMM.update(fg_pixels)
    return bgGMM, fgGMM


def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    min_cut = [[], []]
    energy = 0

    # Build graph
    bgd_idxs, fgd_idxs, pr_bgd_idxs, pr_fgd_idxs = partition_pixels_indexes(img, mask)
    unk_idxs = np.concatenate((pr_bgd_idxs, pr_fgd_idxs))
    rows, cols, _ = img.shape
    src_vertex = rows * cols
    sink_vertex = src_vertex + 1

    src_to_unk_edges = zip([src_vertex] * len(unk_idxs), unk_idxs)
    unk_to_sink_edges = zip(unk_idxs, [sink_vertex] * len(unk_idxs))
    src_to_bgd_edges = zip([src_vertex] * len(bgd_idxs), bgd_idxs)
    bgd_to_sink = zip(bgd_idxs, [sink_vertex] * len(bgd_idxs))
    src_to_fgd_edges = zip([src_vertex] * len(fgd_idxs), fgd_idxs)
    fgd_to_sink_edges = zip(fgd_idxs, [sink_vertex] * len(fgd_idxs))

    src_to_unk_capacities = -np.log(bgGMM.calc_prob(img.reshape(-1, 3)[unk_idxs]))
    unk_to_sink_capcities = []
    src_to_bgd_capacities = itertools.repeat(0, bgd_idxs.size)
    bgd_to_sink_capacities = itertools.repeat(bgGMM.n_components, bgd_idxs.size)
    src_to_fgd_capacities = itertools.repeat(fgGMM.n_components, fgd_idxs.size)
    fgd_to_sink_capacities = itertools.repeat(0, fgd_idxs.size)

    edges = itertools.chain(
        src_to_unk_edges,
        unk_to_sink_edges,
        src_to_bgd_edges,
        bgd_to_sink,
        src_to_fgd_edges,
        fgd_to_sink_edges,
    )

    # t-links
    graph = ig.Graph(n=rows * cols + 2, edges=edges)

    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    convergence = False
    return convergence


def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_name",
        type=str,
        default="banana1",
        help="name of image from the course files",
    )
    parser.add_argument("--eval", type=int, default=1, help="calculate the metrics")
    parser.add_argument(
        "--input_img_path",
        type=str,
        default="",
        help="if you wish to use your own img_path",
    )
    parser.add_argument(
        "--use_file_rect", type=int, default=1, help="Read rect from course files"
    )
    parser.add_argument(
        "--rect",
        type=str,
        default="1,1,100,100",
        help="if you wish change the rect (x,y,w,h",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == "":
        input_path = f"data/imgs/{args.input_name}.jpg"
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(
            map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(" "))
        )
    else:
        rect = tuple(map(int, args.rect.split(",")))

    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f"data/seg_GT/{args.input_name}.bmp", cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f"Accuracy={acc}, Jaccard={jac}")

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow("Original Image", img)
    cv2.imshow("GrabCut Mask", 255 * mask)
    cv2.imshow("GrabCut Result", img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
