# pylint: disable=missing-function-docstring
# pylint: disable=missing-function-docstring, no-member
import cv2
import numpy as np

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

    def bilinear_interpolate(self, x, y, c):
        x1 = int(np.floor(x))
        y1 = int(np.floor(y))
        x2 = min(x1 + 1, self.w - 1)
        y2 = min(y1 + 1, self.h - 1)

        dx = x - x1
        dy = y - y1

        # Get the four surrounding pixels
        Q11 = self.get_pixel(x1, y1, c)  # top-left
        Q12 = self.get_pixel(x2, y1, c)  # top-right
        Q21 = self.get_pixel(x1, y2, c)  # bottom-left
        Q22 = self.get_pixel(x2, y2, c)  # bottom-right

        # Interpolate in x direction (top and bottom rows)
        R1 = (1 - dx) * Q11 + dx * Q12  # top row
        R2 = (1 - dx) * Q21 + dx * Q22  # bottom row

        # Interpolate in y direction between R1 and R2
        return (1 - dy) * R1 + dy * R2

    def bilinear_resize(self, w_new, h_new):
        im_new = Image(w_new, h_new, self.c)

        for c in range(self.c):
            for y in range(h_new):
                for x in range(w_new):
                    # Map back to original image coordinates
                    x_orig = (x + 0.5) * self.w / w_new - 0.5
                    y_orig = (y + 0.5) * self.h / h_new - 0.5

                    # Clamp x_orig and y_orig to ensure we stay within bounds
                    x_orig = max(0, min(self.w - 1, x_orig))
                    y_orig = max(0, min(self.h - 1, y_orig))

                    val = self.bilinear_interpolate(x_orig, y_orig, c)
                    im_new.set_pixel(x, y, c, val)

        return im_new
    def l1_normalize(self):
        total = np.sum(self.data)
        if total != 0:
            self.data /= total

    @classmethod
    def make_box_filter(cls, w):
        filter_img = cls(w, w, 1)  # single channel
        filter_img.data[:, :, 0] = 1.0
        filter_img.l1_normalize()
        return filter_img
    
    def convolve(self, filter_img, preserve):
        assert filter_img.c == self.c or filter_img.c == 1, "Filter must have 1 or same number of channels"

        # Determine output channels
        out_c = self.c if preserve else 1
        out = Image(self.w, self.h, out_c)

        fw, fh = filter_img.w, filter_img.h
        fx_center = fw // 2
        fy_center = fh // 2

        for y in range(self.h):
            for x in range(self.w):
                for c_out in range(out_c):
                    acc = 0.0
                    for fy in range(fh):
                        for fx in range(fw):
                            ix = x + fx - fx_center
                            iy = y + fy - fy_center

                            for fc in range(filter_img.c):
                                # Determine which channel to use
                                ic = fc if filter_img.c == self.c else c_out

                                val = self.get_pixel(ix, iy, ic)
                                filt = filter_img.get_pixel(fx, fy, fc)
                                acc += val * filt

                    # If preserve = 0, all channels go to output channel 0
                    out.set_pixel(x, y, c_out if preserve else 0, acc)

        return out
    @classmethod
    def make_highpass_filter(cls):
        filter_img = cls(3, 3, 1)
        filter_img.data[:, :, 0] = np.array([
            [ 0, -1,  0],
            [-1,  4, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
        return filter_img

    @classmethod
    def make_sharpen_filter(cls):
        filter_img = cls(3, 3, 1)
        filter_img.data[:, :, 0] = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
        return filter_img

    @classmethod
    def make_emboss_filter(cls):
        filter_img = cls(3, 3, 1)
        filter_img.data[:, :, 0] = np.array([
            [-2, -1,  0],
            [-1,  1,  1],
            [ 0,  1,  2]
        ], dtype=np.float32)
        return filter_img
    @classmethod
    def make_gaussian_filter(cls, sigma):
        import math

        size = int(np.ceil(6 * sigma))
        if size % 2 == 0:
            size += 1

        filter_img = cls(size, size, 1)
        center = size // 2
        two_sigma_sq = 2 * sigma * sigma
        norm_factor = 1 / (2 * math.pi * sigma * sigma)

        for y in range(size):
            for x in range(size):
                dx = x - center
                dy = y - center
                value = norm_factor * math.exp(-(dx*dx + dy*dy) / two_sigma_sq)
                filter_img.set_pixel(x, y, 0, value)

        filter_img.l1_normalize()
        return filter_img
    @classmethod
    def make_gx_filter(cls):
        filter_img = cls(3, 3, 1)
        gx_values = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ]
        for y in range(3):
            for x in range(3):
                filter_img.set_pixel(x, y, 0, gx_values[y][x])
        return filter_img
    @classmethod
    def make_gy_filter(cls):
        filter_img = cls(3, 3, 1)
        gy_values = [
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1],
        ]
        for y in range(3):
            for x in range(3):
                filter_img.set_pixel(x, y, 0, gy_values[y][x])
        return filter_img
    def feature_normalize(self):
        min_val = np.min(self.data)
        max_val = np.max(self.data)
        range_val = max_val - min_val

        if range_val == 0:
            self.data[:] = 0
        else:
            self.data = (self.data - min_val) / range_val

    def sobel_image(self):
        gx_filter = Image.make_gx_filter()
        gy_filter = Image.make_gy_filter()

        gx = im.convolve(gx_filter, preserve=0)
        gy = im.convolve(gy_filter, preserve=0)

        mag = Image(im.w, im.h, 1)
        direction = Image(im.w, im.h, 1)

        for y in range(im.h):
            for x in range(im.w):
                g_x = gx.get_pixel(x, y, 0)
                g_y = gy.get_pixel(x, y, 0)

                mag_val = (g_x**2 + g_y**2) ** 0.5
                theta = np.arctan2(g_y, g_x)

                mag.set_pixel(x, y, 0, mag_val)
                direction.set_pixel(x, y, 0, theta)

        return mag, direction


im = Image.load("tung.jpg")  # Load your image

# Apply Sobel operator
mag, direction = im.sobel_image()  # Call as an instance method (using 'im')

# Optionally normalize the images
mag.feature_normalize()  # Use the feature_normalize method, not normalize
direction.feature_normalize()  # Same for direction

# Save the results
mag.save("sobel_magnitude.png")
direction.save("sobel_direction.png")

# Optionally display the images
cv2.imshow("Gradient Magnitude", mag.data)
cv2.imshow("Gradient Direction", direction.data)
cv2.waitKey(0)
cv2.destroyAllWindows()
