# Import libraries
import sys
import os
import math
import tkinter as tk
import tkinter.filedialog
import tkinter.simpledialog
import tkinter.font
import numpy as np
from scipy.spatial import KDTree
from skimage.color import rgb2lab
from skimage.color import lab2rgb

# Commented out so that sklearn library is not needed
# from sklearn.cluster import KMeans

################################
# Draw Image
################################

# Puts a single pixel into Tkinter image
def draw_pixel(canvas_image, pixel, x, y):
    canvas_image.put(hex(pixel), (x, y))

# Draws an RGB image to a Tkinter canvas image
def draw_image(width, height, image):
    # Create empty Tkinter image with correct dimensions
    canvas_image = tk.PhotoImage(width=width, height=height)

    # Draw each pixel
    for y, row in enumerate(image):
        for x, pixel in enumerate(row):
            draw_pixel(canvas_image, pixel, x, y)

    return canvas_image

################################
# Color Conversion
################################

# Convert integer RGB values [0,255] to a hex color code string
def hex(pixel):
    (r, g, b) = pixel
    return '#{0:02x}{1:02x}{2:02x}'.format(r, g, b)

# Converts an image from RGB to YUV
def rgb_image_to_luminance(image):
    (height, width, _) = image.shape

    # RGB in range [0,255]
    pixels = image.reshape(height * width, 3)

    # RGB in range [0-1]
    pixels = pixels / 255
    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]

    # Y in range [0,1]
    y = 0.299*r + 0.587*g + 0.114*b

    # YUV in range [0,255]
    y = y * 255

    y_image = y.reshape(height, width)
    return np.round(y_image)

################################
# Discrete Cosine Transform
################################

# Creates the n-by-n DCT matrix
def create_dct_matrix(n):
    matrix = np.zeros(shape=(n, n), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            a = math.sqrt(1 / n) if i == 0 else math.sqrt(2 / n)
            matrix[i][j] = a * math.cos(((2 * j + 1) * i * math.pi) / (2 * n))

    return matrix

# Computes the 2D DCT for each n x n submatrix of a matrix, in place
def dct2d(matrix, n):
    dct_matrix = create_dct_matrix(n)
    dct_transpose_matrix = np.transpose(dct_matrix)
    (height, width) = matrix.shape

    for y in range(height // n):
        for x in range(width // n):
            block = matrix[n*y:n*y+n, n*x:n*x+n]

            block = np.matmul(dct_matrix, np.matmul(block, dct_transpose_matrix))

            matrix[n*y:n*y+n, n*x:n*x+n] = block

    return matrix

################################
# Quantization
################################

def create_quantization_matrix(n):
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == (n-1) and j == (n-1):
                matrix[n-1][n-1] = 1
            else:
                minimum = min(i, j)
                matrix[i][j] = 2 ** (n - minimum - 2)

    return matrix

# Applies quantization to each block
def quantization(matrix, n):
    quantization_matrix = create_quantization_matrix(n)
    (height, width) = matrix.shape

    for y in range(height // n):
        for x in range(width // n):
            block = matrix[n*y:n*y+n, n*x:n*x+n]

            block = np.round(block / quantization_matrix)

            matrix[n*y:n*y+n, n*x:n*x+n] = block

    return matrix

################################
# Weighting
################################

def weighting(matrix, n):
    (height, width) = matrix.shape

    vertical_blocks = height // n
    horizontal_blocks = width // n

    weights = np.zeros((vertical_blocks, horizontal_blocks))

    for y in range(vertical_blocks):
        for x in range(horizontal_blocks):
            # weights[y][x] = np.sum(matrix[n*y:n*y+n, n*x:n*x+n] ** 2)
            weights[y][x] = np.sum(np.absolute(matrix[n*y:n*y+n, n*x:n*x+n]))

    return weights

################################
# Read and Save Files
################################

# Returns the file name to read
def get_file_name():
    return tk.filedialog.askopenfilename()

# Returns the file extension of a file name
def get_file_extension(file_name):
    return file_name[file_name.rindex('.'):]

# Returns the file name without the extension
def get_file_name_without_extension(file_name):
    return file_name[:file_name.rindex('.')]

# Returns the file size of an open file
def get_file_size(file):
    return os.fstat(f.fileno()).st_size

# Returns the size of a file
def get_file_size(file_name):
    return os.stat(file_name).st_size

def save_bitmap_file(file_name, image, num_points):
    # Get header of original bitmap file, which will be unchanged in the saved file
    file = open(file_name, 'rb')
    header = file.read(54)
    file.close()

    file_name_no_extension = get_file_name_without_extension(file_name)
    height, width, _ = image.shape

    file = open(file_name_no_extension + '_lowpoly_' + str(num_points) + '.bmp', 'wb')
    file.write(header)

    # Compute the number of padding bytes at the end of each row
    row_padding = (math.ceil(width * 3 / 4) * 4) - (width * 3)

    flipped_image = np.flipud(image)

    for row in flipped_image:
        for [r, g, b] in row:
            file.write(int(b).to_bytes(1, byteorder='little', signed=False))
            file.write(int(g).to_bytes(1, byteorder='little', signed=False))
            file.write(int(r).to_bytes(1, byteorder='little', signed=False))

        for p in range(row_padding):
            file.write(b'\x00')

    file.close()

    return

# Reads the bitmap file
# Returns a numpy array of the pixels with shape (height, width, 3)
def read_bitmap_file(file_name):
    file = open(file_name, 'rb')

    # Check that the file is a bitmap file
    signature = file.read(2).decode('ascii') # Get Signature
    if signature != 'BM':
        sys.exit('ERROR: {} is not a bitmap file'.format(file_name))

    file_size = int.from_bytes(file.read(4), 'little') # Get FileSize
    file.read(4) # Discard reserved
    data_offset = int.from_bytes(file.read(4), 'little') # Get DataOffset

    file.read(4) # Discard Size
    width = int.from_bytes(file.read(4), 'little') # Get Width
    height = int.from_bytes(file.read(4), 'little') # Get Height
    file.read(2) # Discard Planes

    # Check that the file is 24 bits per pixel
    bits_per_pixel = int.from_bytes(file.read(2), 'little') # Get BitsPerPixel
    if bits_per_pixel != 24:
        sys.exit('ERROR: file must have 24 bits per pixel')

    # Check that the file is uncompressed
    compression = int.from_bytes(file.read(4), 'little') # Get Compression
    if compression != 0:
        sys.exit('ERROR: file must be uncompressed')

    # Discard the rest of the header
    file.read(data_offset - 34)

    # Create numpy array to hold pixel values
    image = np.zeros(shape=(height, width, 3), dtype=np.float64)

    # Coordinates of current pixel being read/drawn
    x = 0
    y = height - 1

    # Compute the number of padding bytes at the end of each row
    row_padding = (math.ceil(width * 3 / 4) * 4) - (width * 3)

    # Read the full image from the file
    while True:
        # Read each channel value of the next pixel
        blue = file.read(1)
        green = file.read(1)
        red = file.read(1)

        # End of file reached
        if not (blue and green and red):
            break

        # Convert bytes to integers
        blue = int.from_bytes(blue, 'little')
        green = int.from_bytes(green, 'little')
        red = int.from_bytes(red, 'little')

        # Save pixel in image array
        image[y][x] = (red, green, blue)

        x += 1
        if x == width:
            x = 0
            y -= 1
            file.read(row_padding) # Discard padding at end of row

    file.close()
    return image

################################
# Voronoi Diagram
################################

# Generate random points for the Voronoi
# Not used in the final implentation
def generate_points(height, width, num_points):
    points = np.stack([np.random.randint(0, height, num_points), np.random.randint(0, width, num_points)], axis=1)
    return points

# Generate the Voronoi diagram, given a list of Voronoi points
def voronoi_diagram(image, points):
    (height, width, _) = image.shape
    (num_points, _) = points.shape
    tree = KDTree(points)
    voronoi_image = np.zeros((height, width), dtype=np.int64)
    regions = [[] for i in range(num_points)]

    for i, row in enumerate(voronoi_image):
        for j, entry in enumerate(row):
            site_index = tree.query([i, j])[1]
            voronoi_image[i][j] = site_index
            regions[site_index].append(image[i][j])

    final_regions = []
    for region in regions:
        final_regions.append(np.array(region))

    return voronoi_image, final_regions

################################
# Color Selection
################################

# Computes a random color for each Voronoi point
# Not used in the final implementation
def compute_random_colors(regions):
    num_points = len(regions)
    colors = np.zeros((num_points, 3))

    for i, region in enumerate(regions):
        k = np.random.randint(0, len(region))
        colors[i] = region[k]
    return colors

# Computes the mean RGB color of each region
# Not used in the final implementation
def compute_average_rgb_colors(regions):
    num_points = len(regions)
    colors = np.zeros((num_points, 3))

    for i, region in enumerate(regions):
        r = round(np.average(region[:, 0]))
        g = round(np.average(region[:, 1]))
        b = round(np.average(region[:, 2]))
        colors[i] = [r, g, b]

    return colors

# Computes the root mean square RGB color of each region
# Not used in the final implementation
def compute_root_mean_square_rgb_colors(regions):
    num_points = len(regions)
    colors = np.zeros((num_points, 3))

    for i, region in enumerate(regions):
        r = round(np.sqrt(np.average(region[:, 0] ** 2)))
        g = round(np.sqrt(np.average(region[:, 1] ** 2)))
        b = round(np.sqrt(np.average(region[:, 2] ** 2)))
        colors[i] = [r, g, b]

    return colors

# Computes the mean Lab color of each region
def compute_average_lab_colors(regions):
    num_points = len(regions)
    colors = np.zeros((num_points, 3))

    for i, region in enumerate(regions):
        region = region / 255
        lab_region = rgb2lab(region)
        l = np.average(lab_region[:, 0])
        a = np.average(lab_region[:, 1])
        b = np.average(lab_region[:, 2])
        [r, g, b] = lab2rgb([l, a, b])

        colors[i] = [round(r * 255), round(g * 255), round(b * 255)]

    return colors

# Computes the dominant color of each region using k-means clustering
# Not used in the final implementation
# def compute_kmean_dominant_colors(regions):
#     num_points = len(regions)
#     colors = np.zeros((num_points, 3))
#
#     for i, region in enumerate(regions):
#         try:
#             clustering = KMeans(n_clusters=3)
#             clustering.fit(region)
#         except ValueError:
#             clustering = KMeans(n_clusters=1)
#             clustering.fit(region)
#         dominant_color = clustering.cluster_centers_[0]
#         colors[i] = [int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2])]
#
#     return colors

################################
# Operations
################################

def voronoi_points_from_weights(weights, num_points, block_size):
    (vertical_blocks, horizontal_blocks) = weights.shape

    coordinates = np.indices((vertical_blocks, horizontal_blocks)).transpose(1, 2, 0).reshape(-1, 2)
    indices = np.arange(0, vertical_blocks * horizontal_blocks)

    points = np.zeros((num_points, 2))
    total = np.sum(weights)
    probabilities = weights / total
    probabilities = probabilities.reshape(-1)

    for k in range(num_points):
        random_index = np.random.choice(indices, p=probabilities)
        [y, x] = coordinates[random_index]
        i = np.random.randint(0, block_size)
        j = np.random.randint(0, block_size)
        points[k] = [y*block_size+i, x*block_size+j]

    unique_points = np.unique(points, axis=0)
    new_num_points = unique_points.shape[0]

    return unique_points, new_num_points

# Creates an image using a Voronoi diagram and colors selected for each reason
def create_polygon_image(voronoi_image, colors):
    (height, width) = voronoi_image.shape
    image = np.zeros((height, width, 3))

    for i, row in enumerate(voronoi_image):
        for j, entry in enumerate(row):
            color = colors[entry]
            image[i][j] = color

    image = image.astype(np.uint8)
    return image

################################
# Main Program
################################
def main():
    # Create the display window
    window = tk.Tk()
    window.title('Loading...')

    # Get the file and read it
    file_name = get_file_name()

    # Get the number of Voronoi points to use (equal to the number of polygons)
    num_points = quality = tk.simpledialog.askstring(title='Voronoi Points', prompt='Enter the integer number of polygons to use:')

    try:
        int(num_points)
    except ValueError:
        print('ERROR: not a valid number')
        sys.exit(1)

    num_points = int(num_points)

    if num_points < 0:
        print('ERROR: invalid number of points')

    # Read the bitmap image file
    original_image = read_bitmap_file(file_name)
    (height, width, _) = original_image.shape

    # The block size for the 2D DCT
    block_size = 8

    # Compute the luminance (Y channel) from RGB image
    luminance = rgb_image_to_luminance(original_image)

    # Apply DCT to the luminance channel
    dct_image = dct2d(luminance, block_size)

    quantized_image = quantization(dct_image, block_size)

    weighted_image = weighting(quantized_image, block_size)

    initial_num_points = num_points

    # Select Voronoi points based on the weighted distribution
    points, num_points = voronoi_points_from_weights(weighted_image, num_points, block_size)

    # Generate Voronoi points randomly
    # points = generate_points(height, width, num_points)

    # Create the Voronoi diagram and get all polygon regions (pixel sets)
    voronoi_image, regions = voronoi_diagram(original_image, points)

    # Get the average Lab color for each polygon region
    colors = compute_average_lab_colors(regions)

    # colors = compute_random_colors(regions)
    # colors = compute_average_rgb_colors(regions)
    # colors = compute_root_mean_square_rgb_colors(regions)
    # colors = compute_kmean_dominant_colors(regions)

    # Generate the low polygon image using the Voronoi diagram and selected colors
    image = create_polygon_image(voronoi_image, colors)

    # Save the low-polygon image to a new bitmap file
    save_bitmap_file(file_name, image, initial_num_points)

    # Display the original image and low poly image side by side
    buffer_width = 64
    canvas = tk.Canvas(window, width=width*2+buffer_width, height=height)
    original_image = original_image.astype(np.uint8)
    canvas_original_image = draw_image(width, height, original_image)
    canvas_voronoi_image = draw_image(width, height, image)
    canvas.create_image(0, 0, image=canvas_original_image, state='normal', anchor='nw')
    canvas.create_image(width+buffer_width, 0, image=canvas_voronoi_image, state='normal', anchor='nw')
    canvas.pack()

    # Update the window
    window.mainloop()


    return

if __name__ == '__main__':
    main()
