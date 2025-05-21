# pylint: disable=missing-function-docstring
# pylint: disable=missing-function-docstring, no-member
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import math
import random

@dataclass
class Match:
    a: int  # index of descriptor in image A
    b: int  # index of best matching descriptor in image B
    distance: float
class Image:
    def __init__(self, w, h, c):
        self.w = w  # width
        self.h = h  # height
        self.c = c  # number of channels
        self.data = np.zeros((h, w, c), dtype=np.float32)  # HWC format

    @classmethod
    def load(cls, filename):
        bgr_img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise ValueError(f"Image {filename} could not be loaded.")
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        np_img = rgb_img.astype(np.float32) / 255.0  # normalize to [0,1]
        h, w, c = np_img.shape
        img = cls(w, h, c)
        img.data = np_img  # store in HWC format directly
        return img

    def get_pixel(self, x, y, c):
        # Clamp coordinates
        x = min(max(x, 0), self.w - 1)
        y = min(max(y, 0), self.h - 1)
        c = min(max(c, 0), self.c - 1)
        return self.data[y, x, c]

    def set_pixel(self, x, y, c, value):
        self.data[y, x, c] = value

    def nn_interpolate(self, x, y, c):
        ix = round(x)
        iy = round(y)
        return self.get_pixel(ix, iy, c)

    def nn_resize(self, w_new, h_new):
        im_new = Image(w_new, h_new, self.c)

        for c in range(self.c):
            for y in range(h_new):
                for x in range(w_new):
                    # Map back to original image coordinates
                    x_orig = (x + 0.5) * self.w / w_new - 0.5
                    y_orig = (y + 0.5) * self.h / h_new - 0.5

                    val = self.nn_interpolate(x_orig, y_orig, c)
                    im_new.set_pixel(x, y, c, val)

        return im_new
    def save(self, filename):
        bgr_img = cv2.cvtColor((self.data * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr_img)

    def structure_matrix(self, sigma):
        gray = np.mean(self.data, axis=2)  # Convert to grayscale

        # Compute gradients
        Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        # Compute products of derivatives
        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        Ixy = Ix * Iy

        # Apply Gaussian blur (weighted sum of local values)
        Sxx = cv2.GaussianBlur(Ix2, (0, 0), sigma)
        Syy = cv2.GaussianBlur(Iy2, (0, 0), sigma)
        Sxy = cv2.GaussianBlur(Ixy, (0, 0), sigma)

        return Sxx, Syy, Sxy
    def make_1d_gaussian(self, sigma):
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1  # ensure odd size
        center = size // 2
        x = np.arange(size) - center
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel /= kernel.sum()  # normalize
        return kernel
    def smooth_image(self, sigma):
        kernel_1d = self.make_1d_gaussian(sigma)

        # Expand to horizontal kernel (1 x N)
        kernel_x = kernel_1d.reshape(1, -1)
        # Expand to vertical kernel (N x 1)
        kernel_y = kernel_1d.reshape(-1, 1)

        # Allocate output
        smoothed = np.zeros_like(self.data)

        # Apply to each channel
        for c in range(self.c):
            channel = self.data[:, :, c]
            blurred_x = cv2.filter2D(channel, -1, kernel_x, borderType=cv2.BORDER_REFLECT)
            blurred_y = cv2.filter2D(blurred_x, -1, kernel_y, borderType=cv2.BORDER_REFLECT)
            smoothed[:, :, c] = blurred_y

        result = Image(self.w, self.h, self.c)
        result.data = smoothed
        return result
    def cornerness_response(self, Sxx, Syy, Sxy, alpha=0.06):
        # Harris corner response: R = det(M) - alpha * trace(M)^2
        det_M = Sxx * Syy - Sxy**2
        trace_M = Sxx + Syy
        R = det_M - alpha * (trace_M ** 2)

        # Wrap in an Image object
        result = Image(self.w, self.h, 1)
        result.data = R[:, :, np.newaxis]  # Convert to shape (h, w, 1)
        return result
    def nms_image(self, w):
        assert self.c == 1, "NMS typically works on single-channel response images"

        # Copy original data to avoid modifying in-place
        nms_data = self.data.copy()
        h, w_img = self.h, self.w

        # Pad the array to simplify boundary handling
        pad = w
        padded = np.pad(nms_data[:, :, 0], pad, mode='constant', constant_values=-np.inf)

        for y in range(h):
            for x in range(w_img):
                # Extract window around (x,y) in padded image
                window = padded[y:y + 2 * pad + 1, x:x + 2 * pad + 1]
                center_val = padded[y + pad, x + pad]

                # If any neighbor (other than center) has a greater value, suppress this pixel
                if np.any(window > center_val):
                    nms_data[y, x, 0] = -1e10  # very low negative number

        result = Image(self.w, self.h, self.c)
        result.data = nms_data
        return result
    def harris_corner_detector(self, sigma, thresh, nms, n_ptr):
        # Step 1: Compute structure matrix components
        Sxx, Syy, Sxy = self.structure_matrix(sigma)

        # Step 2: Compute cornerness response
        R = self.cornerness_response(Sxx, Syy, Sxy)

        # Step 3: Non-maximum suppression (NMS)
        R_nms = R.nms_image(nms)

        # Step 4: Threshold and extract keypoints
        keypoints = []
        h, w = R.h, R.w
        for y in range(h):
            for x in range(w):
                if R_nms.data[y, x, 0] > thresh:
                    keypoints.append({
                        'x': x,
                        'y': y,
                        'response': R_nms.data[y, x, 0]
                    })
        # Step 5: Set number of corners found
        n_ptr[0] = len(keypoints)

        # Step 6: Return list of descriptors (keypoints)
        return keypoints
    def draw_corners(self, corners, color=(1.0, 0.0, 0.0), size=3):
        img_copy = self.data.copy()

        for corner in corners:
            x, y = int(corner['x']), int(corner['y'])
            for dx in range(-size, size + 1):
                if 0 <= x + dx < self.w and 0 <= y < self.h:
                    img_copy[y, x + dx] = color
                if 0 <= x < self.w and 0 <= y + dx < self.h:
                    img_copy[y + dx, x] = color

        result = Image(self.w, self.h, self.c)
        result.data = img_copy
        return result
    @staticmethod
    def l1_distance(a, b):
        assert a.shape == b.shape, "Vectors must be of the same shape"
        return np.sum(np.abs(a - b))
    @staticmethod
    def match_descriptors(desc_a, desc_b):
        matches = []

        for i, da in enumerate(desc_a):
            best_index = -1
            best_distance = float('inf')
            for j, db in enumerate(desc_b):
                dist = Image.l1_distance(da.data, db.data)
                if dist < best_distance:
                    best_distance = dist
                    best_index = j
            matches.append(Match(a=i, b=best_index, distance=best_distance))

        return matches
    @staticmethod
    def remove_duplicate_matches(matches):
        # Sort matches by distance (best matches first)
        matches.sort(key=lambda m: m.distance)

        used_b = set()
        unique_matches = []

        for match in matches:
            if match.b not in used_b:
                unique_matches.append(match)
                used_b.add(match.b)

        return unique_matches
    @staticmethod
    def project_point(H, p):
        """
        Project point p = (x, y) using homography H.
        H is a 3x3 NumPy array.
        p is a tuple or list: (x, y)
        Returns projected point (x_proj, y_proj)
        """
        x, y = p
        pt = np.array([x, y, 1.0])
        projected = H @ pt  # matrix multiplication
        x_proj = projected[0] / projected[2]
        y_proj = projected[1] / projected[2]
        return (x_proj, y_proj)
    @staticmethod
    def point_distance(p, q):
        """
        Compute Euclidean distance between points p and q.
        Each is a tuple or list of (x, y).
        """
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        return math.sqrt(dx * dx + dy * dy)
    @staticmethod
    def model_inliers(H, matches, desc_a, desc_b, thresh):
        """
        Given a homography H, a list of matches, and their corresponding descriptors,
        return the number of inliers and reorder the list in-place to have inliers at front.

        Parameters:
        - H: 3x3 NumPy array (homography)
        - matches: list of Match(a=idx_in_A, b=idx_in_B, distance=...)
        - desc_a: list of descriptors from image A (each has .p = (x, y))
        - desc_b: list of descriptors from image B (each has .p = (x, y))
        - thresh: maximum distance for a match to be considered an inlier

        Returns:
        - int: number of inliers
        """
        inliers = 0
        for i in range(len(matches)):
            match = matches[i]
            pa = desc_a[match.a].p  # point from image A
            pb = desc_b[match.b].p  # point from image B

            projected = Image.project_point(H, pa)
            dist = Image.point_distance(projected, pb)

            if dist < thresh:
                # Swap current match to the inlier section at the front
                matches[i], matches[inliers] = matches[inliers], matches[i]
                inliers += 1

        return inliers
    @staticmethod
    def randomize_matches(matches):
        for i in range(len(matches) - 1, 0, -1):
            j = random.randint(0, i)
            matches[i], matches[j] = matches[j], matches[i]
    @staticmethod
    def compute_homography(matches, keypoints_a, keypoints_b):
        assert len(matches) >= 4, "Need at least 4 matches to compute homography."

        A = []
        for match in matches:
            pa = keypoints_a[match.a]
            pb = keypoints_b[match.b]
            x, y = pa['x'], pa['y']
            xp, yp = pb['x'], pb['y']

            A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
            A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)

        return H / H[2, 2]
    @staticmethod
    def RANSAC(matches, keypoints_a, keypoints_b, thresh=3.0, k=1000, cutoff=10):
        best_H = None
        best_inliers = []

        for _ in range(k):
            # Step 1: Random subset
            subset = random.sample(matches, 4)
            H = Image.compute_homography(subset, keypoints_a, keypoints_b)

            # Step 2: Count inliers
            inliers = []
            for match in matches:
                pa = keypoints_a[match.a]
                pb = keypoints_b[match.b]
                projected = Image.project_point(H, (pa['x'], pa['y']))
                dist = Image.point_distance(projected, (pb['x'], pb['y']))
                if dist < thresh:
                    inliers.append(match)

            # Step 3: Check best
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_H = H
                if len(inliers) > cutoff:
                    break

        # Optional: recompute H with all inliers
        if best_inliers:
            best_H = Image.compute_homography(best_inliers, keypoints_a, keypoints_b)

        return best_H, best_inliers
    @staticmethod
    def bilinear_interpolate(image, x, y, c):
        h, w = image.shape[:2]

        x0 = int(np.floor(x))
        x1 = x0 + 1
        y0 = int(np.floor(y))
        y1 = y0 + 1

        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)

        Ia = image[y0, x0, c]
        Ib = image[y1, x0, c]
        Ic = image[y0, x1, c]
        Id = image[y1, x1, c]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa * Ia + wb * Ib + wc * Ic + wd * Id
    @staticmethod
    def combine_images(a, b, H):
        H_inv = np.linalg.inv(H)

        # Step 1: project b’s corners into a’s coordinate frame
        corners_b = [(0, 0), (b.w - 1, 0), (0, b.h - 1), (b.w - 1, b.h - 1)]
        projected = [Image.project_point(H_inv, pt) for pt in corners_b]

        all_x = [0, a.w - 1] + [x for x, y in projected]
        all_y = [0, a.h - 1] + [y for x, y in projected]

        min_x, max_x = int(np.floor(min(all_x))), int(np.ceil(max(all_x)))
        min_y, max_y = int(np.floor(min(all_y))), int(np.ceil(max(all_y)))

        # Step 2: compute new width/height and translation
        new_w = max_x - min_x + 1
        new_h = max_y - min_y + 1
        offset_x, offset_y = -min_x, -min_y

        # Step 3: allocate new canvas and copy image a
        c = a.c
        canvas = Image(new_w, new_h, c)

        for y in range(a.h):
            for x in range(a.w):
                for ch in range(c):
                    val = a.get_pixel(x, y, ch)
                    canvas.set_pixel(x + offset_x, y + offset_y, ch, val)

        # Step 4: fill in pixels from image b using inverse warping
        for y in range(new_h):
            for x in range(new_w):
                # Transform canvas point (x - offset_x, y - offset_y) to b's space
                src_x, src_y = Image.project_point(H, (x - offset_x, y - offset_y))

                if 0 <= src_x < b.w and 0 <= src_y < b.h:
                    for ch in range(c):
                        val = Image.bilinear_interpolate(b.data, src_x, src_y, ch)
                        canvas.set_pixel(x, y, ch, val)

        return canvas
    def panorama_image(self,im1, im2, thresh=3.0, k=1000, cutoff=10):
        # Detect corners and extract descriptors
        n1, n2 = [0], [0]
        kps1 = im1.harris_corner_detector(sigma=2.0, thresh=0.1, nms=3, n_ptr=n1)
        kps2 = im2.harris_corner_detector(sigma=2.0, thresh=0.1, nms=3, n_ptr=n2)

        # Match descriptors
        matches = Image.match_descriptors(kps1, kps2)
        matches = Image.remove_duplicate_matches(matches)

        # Estimate homography
        H, _ = Image.RANSAC(matches, kps1, kps2, thresh=thresh, k=k, cutoff=cutoff)

        # Combine images
        return Image.combine_images(im1, im2, H)