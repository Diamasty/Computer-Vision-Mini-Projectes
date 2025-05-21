# pylint: disable=missing-function-docstring
# pylint: disable=missing-function-docstring, no-member
import cv2
import numpy as np

class Image:
    def __init__(self, width, height, channels):
        self.w = width
        self.h = height
        self.c = channels
        self.data = np.zeros((channels, height, width), dtype=np.float32)

    @classmethod
    def load(cls, filename):
        bgr_img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise ValueError(f"Image {filename} could not be loaded.")
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        np_img = rgb_img.astype(np.float32) / 255.0
        h, w, c = np_img.shape
        chw_img = np_img.transpose(2, 0, 1)
        img = cls(w, h, c)
        img.data = chw_img
        return img

    def get_pixel(self, x, y, c):
        x = max(0, min(x, self.w - 1))
        y = max(0, min(y, self.h - 1))
        c = max(0, min(c, self.c - 1))
        return self.data[c, y, x]

    def set_pixel(self, x, y, c, value):
        if 0 <= x < self.w and 0 <= y < self.h and 0 <= c < self.c:
            self.data[c, y, x] = value

    def save(self, filename):
        rgb_img = (self.data.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, bgr_img)

    def copy_image(self):
        copy = Image(self.w, self.h, self.c)
        copy.data = np.copy(self.data)
        return copy
    def rgb_to_grayscale(self):
        gray = Image(self.w, self.h, 1)
        for y in range(self.h):
            for x in range(self.w):
                r = self.get_pixel(x, y, 0)
                g = self.get_pixel(x, y, 1)
                b = self.get_pixel(x, y, 2)
                y_prime = 0.299 * r + 0.587 * g + 0.114 * b
                gray.set_pixel(x, y, 0, y_prime)
        return gray
    
    def shift_image(self, c, v):
        # Loop through all the pixels in the specified channel `c`
        for y in range(self.h):
            for x in range(self.w):
                # Get the current pixel value in channel `c`
                pixel_value = self.get_pixel(x, y, c)
                # Add the constant value `v` to the pixel value
                new_pixel_value = pixel_value + v
                # Clip the value to ensure it stays within the valid range [0.0, 1.0]
                new_pixel_value = max(0.0, min(1.0, new_pixel_value))
                # Set the new pixel value back to the image
                self.set_pixel(x, y, c, new_pixel_value)

    def rgb_to_hsv(self):
        hsv = Image(self.w, self.h, 3)  # Create an empty HSV image with 3 channels
        for y in range(self.h):
            for x in range(self.w):
                r = self.get_pixel(x, y, 0)
                g = self.get_pixel(x, y, 1)
                b = self.get_pixel(x, y, 2)

                # Calculate Value (V)
                V = max(r, g, b)
                m = min(r, g, b)
                C = V - m

                # Calculate Saturation (S)
                S = 0 if V == 0 else C / V

                # Calculate Hue (H)
                if C == 0:  # If no chroma, set hue to 0 (gray pixel)
                    H = 0
                else:
                    if V == r:
                        H = (g - b) / C
                    elif V == g:
                        H = 2 + (b - r) / C
                    elif V == b:
                        H = 4 + (r - g) / C

                # Normalize Hue to be within [0, 1)
                H = H / 6.0
                if H < 0:
                    H += 1

                # Set the HSV values to the new image
                hsv.set_pixel(x, y, 0, H)
                hsv.set_pixel(x, y, 1, S)
                hsv.set_pixel(x, y, 2, V)

        return hsv
    def scale_image(self, c, v):
        for y in range(self.h):
            for x in range(self.w):
                pixel_value = self.get_pixel(x, y, c)
                new_pixel_value = pixel_value * v
                self.set_pixel(x, y, c, new_pixel_value)
    def clamp_image(self):
        self.data = np.clip(self.data, 0.0, 1.0)
    def hsv_to_rgb(self):
        rgb = Image(self.w, self.h, 3)
        for y in range(self.h):
            for x in range(self.w):
                H = self.get_pixel(x, y, 0)
                S = self.get_pixel(x, y, 1)
                V = self.get_pixel(x, y, 2)

                C = V * S
                H_prime = H * 6
                X = C * (1 - abs(H_prime % 2 - 1))

                if 0 <= H_prime < 1:
                    r1, g1, b1 = C, X, 0
                elif 1 <= H_prime < 2:
                    r1, g1, b1 = X, C, 0
                elif 2 <= H_prime < 3:
                    r1, g1, b1 = 0, C, X
                elif 3 <= H_prime < 4:
                    r1, g1, b1 = 0, X, C
                elif 4 <= H_prime < 5:
                    r1, g1, b1 = X, 0, C
                elif 5 <= H_prime < 6:
                    r1, g1, b1 = C, 0, X
                else:
                    r1, g1, b1 = 0, 0, 0

                m = V - C
                r, g, b = r1 + m, g1 + m, b1 + m

                rgb.set_pixel(x, y, 0, r)
                rgb.set_pixel(x, y, 1, g)
                rgb.set_pixel(x, y, 2, b)

        return rgb






im = Image.load("7.jpg")
im = im.rgb_to_hsv()
im.scale_image(1, 2)
im.clamp_image()
im = im.hsv_to_rgb()
im.save("dog_scale_saturated.jpg")