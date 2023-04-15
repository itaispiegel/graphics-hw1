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

GAMMA = 50
LAMBDA = 9 * GAMMA

beta = None
n_link_edges = None
n_link_capacities = None


def require_initialization(func):
    def wrapper(self, *args, **kwargs):
        if not self._initialized:
            raise ValueError("Instance not initialized")
        return func(self, *args, **kwargs)

    return wrapper


class Component:
    def __init__(self, data_points, total_points_count, mean=None):
        self.data_points = data_points
        self.mean = mean if mean is not None else np.mean(data_points, axis=0)
        self.covariance_matrix = np.cov(self.data_points, rowvar=False)
        self.covariance_matrix_det = np.linalg.det(self.covariance_matrix)
        self.covariance_matrix_inverse = np.linalg.inv(self.covariance_matrix)
        self.weight = len(data_points) / total_points_count

    def calc_scores(self, X):
        d = X.shape[1]
        diff = X - self.mean
        norm = np.sqrt((2 * np.pi) ** d * self.covariance_matrix_det)
        exponent = -0.5 * np.einsum(
            "ij, ij->i", diff, np.dot(self.covariance_matrix_inverse, diff.T).T
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

    @property
    @require_initialization
    def weights(self):
        return np.array([c.weight for c in self.components])

    @require_initialization
    def calc_probs(self, X):
        probs = [c.calc_scores(X) for c in self.components]
        return np.dot(self.weights, probs)

    @require_initialization
    def assign_points_to_components(self, X):
        probs = np.array([c.calc_scores(X) for c in self.components]).T
        return np.argmax(probs, axis=1)

    @require_initialization
    def update(self, X):
        labels = self.assign_points_to_components(X)
        updated_components = []
        for i in range(self.n_components):
            assigned_pixels = X[np.where(labels == i)]
            component = Component(assigned_pixels, total_points_count=len(X))
            updated_components.append(component)
        self.components = updated_components


def partition_pixels(img, mask):
    bg_pixels = img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    fg_pixels = img[np.logical_or(mask == GC_FGD, mask == GC_PR_FGD)]
    return bg_pixels, fg_pixels


def t_link_edges_and_capacities(img, mask, bgGMM, fgGMM):
    rows, cols, _ = img.shape
    src_vertex = rows * cols
    sink_vertex = src_vertex + 1

    flat_mask = mask.reshape(-1)
    pr_pixels_idxs = np.where(
        np.logical_or(flat_mask == GC_PR_BGD, flat_mask == GC_PR_FGD)
    )[0]
    bgd_pixels_idxs = np.where(flat_mask == GC_BGD)[0]
    fgd_pixels_idxs = np.where(flat_mask == GC_FGD)[0]

    pr_bgd_pixels = img[mask == GC_PR_BGD]
    pr_fgd_pixels = img[mask == GC_PR_FGD]

    t_link_edges = itertools.chain(
        zip(itertools.repeat(src_vertex, pr_pixels_idxs.size), pr_pixels_idxs),
        zip(pr_pixels_idxs, itertools.repeat(sink_vertex, pr_pixels_idxs.size)),
        zip(itertools.repeat(src_vertex, bgd_pixels_idxs.size), bgd_pixels_idxs),
        zip(bgd_pixels_idxs, itertools.repeat(sink_vertex, bgd_pixels_idxs.size)),
        zip(itertools.repeat(src_vertex, fgd_pixels_idxs.size), fgd_pixels_idxs),
        zip(fgd_pixels_idxs, itertools.repeat(sink_vertex, fgd_pixels_idxs.size)),
    )

    t_link_capacities = itertools.chain(
        -np.log(bgGMM.calc_probs(pr_bgd_pixels)),
        -np.log(bgGMM.calc_probs(pr_fgd_pixels)),
        itertools.repeat(0, bgd_pixels_idxs.size),
        itertools.repeat(LAMBDA, bgd_pixels_idxs.size),
        itertools.repeat(LAMBDA, fgd_pixels_idxs.size),
        itertools.repeat(0, fgd_pixels_idxs.size),
    )

    return t_link_edges, t_link_capacities


def calculate_beta(img):
    global beta
    if beta is not None:
        return beta

    beta = 0
    rows, cols, _ = img.shape
    for y in range(rows):
        for x in range(cols):
            color = img[y, x]
            if x > 0:
                diff = color - img[y, x - 1]
                beta += diff.dot(diff)
            if y > 0 and x > 0:
                diff = color - img[y - 1, x - 1]
                beta += diff.dot(diff)
            if y > 0:
                diff = color - img[y - 1, x]
                beta += diff.dot(diff)
            if y > 0 and x < cols - 1:
                diff = color - img[y - 1, x + 1]
                beta += diff.dot(diff)
    return 1.0 / (2 * beta / (4 * cols * rows - 3 * cols - 3 * rows + 2))


def n_link_edges_and_capacities(img):
    global n_link_edges, n_link_capacities
    if n_link_edges is not None and n_link_capacities is not None:
        return n_link_edges, n_link_capacities

    beta = calculate_beta(img)
    (
        img_without_left_col,
        img_without_right_col,
        img_without_left_col_and_top_row,
        img_without_right_col_and_right_row,
        img_without_top_row,
        img_without_bottom_row,
        img_without_right_col_and_top_row,
        img_without_left_col_and_bottom_row,
    ) = (
        img[:, 1:],
        img[:, :-1],
        img[1:, 1:],
        img[:-1, :-1],
        img[1:, :],
        img[:-1, :],
        img[1:, :-1],
        img[:-1, 1:],
    )

    left_diff = img_without_left_col - img_without_right_col
    upleft_diff = img_without_left_col_and_top_row - img_without_right_col_and_right_row
    up_diff = img_without_top_row - img_without_bottom_row
    upright_diff = (
        img_without_right_col_and_top_row - img_without_left_col_and_bottom_row
    )
    left_V = GAMMA * np.exp(-beta * np.sum(np.square(left_diff), axis=2))
    upleft_V = (
        GAMMA / np.sqrt(2) * np.exp(-beta * np.sum(np.square(upleft_diff), axis=2))
    )
    up_V = GAMMA * np.exp(-beta * np.sum(np.square(up_diff), axis=2))
    upright_V = (
        GAMMA / np.sqrt(2) * np.exp(-beta * np.sum(np.square(upright_diff), axis=2))
    )

    n_link_edges = itertools.chain(
        zip(img_without_left_col.reshape(-1), img_without_right_col.reshape(-1)),
        zip(
            img_without_left_col_and_top_row.reshape(-1),
            img_without_right_col_and_right_row.reshape(-1),
        ),
        zip(img_without_top_row.reshape(-1), img_without_bottom_row.reshape(-1)),
        zip(
            img_without_right_col_and_top_row.reshape(-1),
            img_without_left_col_and_bottom_row.reshape(-1),
        ),
    )

    n_link_capacities = np.concatenate(
        (
            left_V.reshape(-1),
            upleft_V.reshape(-1),
            up_V.reshape(-1),
            upright_V.reshape(-1),
        )
    )

    return n_link_edges, n_link_capacities


def construct_graph(img, mask, bgGMM, fgGMM):
    rows, cols, _ = img.shape
    t_link_edges, t_link_capacities = t_link_edges_and_capacities(
        img, mask, bgGMM, fgGMM
    )
    n_link_edges, n_link_capacities = n_link_edges_and_capacities(img)
    edges = itertools.chain(t_link_edges, n_link_edges)
    capacities = itertools.chain(t_link_capacities, n_link_capacities)
    graph = ig.Graph(n=rows * cols + 2, edges=edges)
    graph.es["capacity"] = capacities
    return graph


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

    graph = construct_graph(img, mask, bgGMM, fgGMM)
    src, sink = graph.vcount() - 2, graph.vcount() - 1
    ig_mincut = graph.mincut(src, sink)
    min_cut = ig_mincut.partition

    return min_cut, energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step
    flat_mask = mask.reshape(-1)
    pr_idxs = np.where(np.logical_or(flat_mask == GC_PR_BGD, flat_mask == GC_PR_FGD))

    flat_mask[pr_idxs] = np.where(
        np.isin(pr_idxs, mincut_sets[0]), GC_PR_FGD, GC_PR_BGD
    )
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
