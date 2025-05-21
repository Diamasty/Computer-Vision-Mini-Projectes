# Load image and test
# im = Image.load("7.jpg")
# print("Pixel (2,1) in R channel before:", im.get_pixel(2, 1, 0))

# im.set_pixel(2, 1, 0, 1.0)
# print("Pixel (2,1) in R channel after:", im.get_pixel(2, 1, 0))

# im.save("output.jpg")

# im = Image.load("7.jpg")
# for y in range(im.h):
#     for x in range(im.w):
#         im.set_pixel(x, y, 0, 0.0)  # Remove red channel
# im.save("7_no_red.jpg")

# im = Image.load("7.jpg")         # Load original image
# copy = im.copy_image()          # Create a copy
# copy.save("7_copy.jpg")          # Save the copy