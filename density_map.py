import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import argparse
from matplotlib import cm as CM
from PIL import Image
import scipy
import scipy.io as io
from scipy import spatial
from scipy.ndimage import gaussian_filter
import h5py

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for density_map', add_help=False)

    parser.add_argument('--images_path', default='')
    
    parser.add_argument('--points_path', default='')

    parser.add_argument('--method', default='fixed',
                        help='fixed , knn')
    
    parser.add_argument('--output_dir', default='',
                        help='path where to save')

    return parser


def generate_density_map_with_fixed_kernel(images_path, all_points, density_map_dir):
    '''
    img: input image.
    all_points: list/array of predicted points per image.
    density_map_dir: path where heatmaps are saved.
    '''
    def gaussian_kernel(size, sigma):
        rows = size[0]
        cols = size[1]
        mean_x = int((rows - 1) / 2)
        mean_y = int((cols - 1) / 2)

        f = np.zeros(size)
        for x in range(0, rows):
            for y in range(0, cols):
                mean_x2 = (x - mean_x) * (x - mean_x)
                mean_y2 = (y - mean_y) * (y - mean_y)
                f[x, y] = (1.0 / (2.0 * np.pi * sigma * sigma)) * np.exp((mean_x2 + mean_y2) / (-2.0 * sigma * sigma))
        return f

    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(valid_exts) and os.path.getsize(os.path.join(images_path, f)) >= 100])

    for idx, fname in enumerate(image_files):
        if idx < len(all_points):
            pts = all_points[idx]
        else:
            pts = []

        points = []
        for p in pts:
            points.append([p[1], p[0]])  # convert (col, row) to (row, col)

        img = plt.imread(os.path.join(images_path, fname))
        rows, cols = img.shape[0], img.shape[1]

        if rows < 600:
            kernel_size, sigma = 200, 30
        elif 600 <= rows < 1200:
            kernel_size, sigma = 300, 40
        else:
            kernel_size, sigma = 400, 60

        # ensure odd kernel size
        if kernel_size % 2 == 0:
            kernel_size += 1

        d_map = np.zeros([rows, cols])

        if len(points) > 0:
            f = gaussian_kernel([kernel_size, kernel_size], sigma)
            normed_f = (1.0 / f.sum()) * f
            half_k = int((f.shape[0] - 1) / 2)

            for p in points:
                r, c = int(p[0]), int(p[1])
                if r >= rows or c >= cols or r < 0 or c < 0:
                    continue

                r_min = max(0, r - half_k)
                r_max = min(rows, r + half_k + 1)
                c_min = max(0, c - half_k)
                c_max = min(cols, c + half_k + 1)

                kf_r_min = half_k - (r - r_min)
                kf_r_max = half_k + (r_max - r)
                kf_c_min = half_k - (c - c_min)
                kf_c_max = half_k + (c_max - c)

                d_map[r_min:r_max, c_min:c_max] += normed_f[kf_r_min:kf_r_max, kf_c_min:kf_c_max]

        filename_base = os.path.splitext(fname)[0]
        out_path = os.path.join(density_map_dir, f'density_map_{filename_base}.png')
        plt.imsave(out_path, d_map, cmap=CM.jet)
        print(f"Generated heatmap for {fname} -> {out_path}")


def gaussian_filter_density(images_path, all_points, density_map_dir):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = sorted([f for f in os.listdir(images_path) if f.lower().endswith(valid_exts) and os.path.getsize(os.path.join(images_path, f)) >= 100])

    for idx, fname in enumerate(image_files):
        if idx < len(all_points):
            points = list(all_points[idx])
        else:
            points = []

        img = plt.imread(os.path.join(images_path, fname))
        img_shape = [img.shape[0], img.shape[1]]
        density = np.zeros(img_shape, dtype=np.float32)
        gt_count = len(points)

        filename_base = os.path.splitext(fname)[0]
        out_path = os.path.join(density_map_dir, f'density_map_{filename_base}.png')

        if gt_count == 0:
            plt.imsave(out_path, density, cmap=CM.jet)
            continue

        leafsize = 2048
        tree = scipy.spatial.KDTree(np.array(points).copy(), leafsize=leafsize)
        k_val = min(4, gt_count)
        distances, locations = tree.query(points, k=k_val)

        for i, pt in enumerate(points):
            pt2d = np.zeros(img_shape, dtype=np.float32)
            if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
                pt2d[int(pt[1]), int(pt[0])] = 1.0
            else:
                continue

            if gt_count > 3:
                sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
            elif gt_count > 1:
                sigma = float(np.mean(distances[i][1:])) * 0.1
            else:
                sigma = float(np.average(np.array(img_shape))) / 4.0

            density += scipy.ndimage.gaussian_filter(pt2d, sigma, mode='constant')

        plt.imsave(out_path, density, cmap=CM.jet)
        print(f"Generated KNN density map for {fname} -> {out_path}")
    

# test code
if __name__=="__main__":
    # show an example to use function generate_density_map_with_fixed_kernel.
    parser = argparse.ArgumentParser('density map', parents=[get_args_parser()])
    args = parser.parse_args()

    imgs_path = args.images_path
    points_path = args.points_path
    if os.path.isdir(points_path):
        candidate = os.path.join(points_path, 'points.npy')
        if os.path.exists(candidate):
            points_path = candidate
    elif points_path.endswith('.txt'):
        candidate = os.path.join(os.path.dirname(points_path), 'points.npy')
        if os.path.exists(candidate):
            points_path = candidate

    All_points = np.load(points_path, allow_pickle=True)
    method = args.method
    density_dir = args.output_dir
    #kernel_size=args.Kernel_size
    #sigma=args.Sigma

    if not os.path.isdir(density_dir):
          os.makedirs(density_dir)

    if method=='fixed':
      generate_density_map_with_fixed_kernel(imgs_path,All_points,density_dir)
    elif method=='knn':
      gaussian_filter_density(imgs_path,All_points,density_dir)