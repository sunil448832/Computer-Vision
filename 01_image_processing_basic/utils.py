import numpy as np


def convolve2D(image, kernel):
    height, width = image.shape
    height_kernel, width_kernel = kernel.shape
    temp_image = np.zeros([height+height_kernel, width+width_kernel])
    temp_image[height_kernel//2:height+height_kernel//2,
               width_kernel//2:width+width_kernel//2] = image
    output = np.zeros(image.shape)
    for i in range(height):
        for j in range(width):
            output[i, j] = np.multiply(
                temp_image[i:i+height_kernel, j:j+width_kernel], kernel).sum()
    return output


def convolve3D(image, kernel):
    image[:, :, 0], image[:, :, 1], image[:, :, 2] = convolve2D(image[:, :, 0], kernel),\
        convolve2D(image[:, :, 1], kernel),\
        convolve2D(image[:, :, 2], kernel)
    return image


def gaussian_distribution(
    x, y, sigma): return np.exp(-(x**2+y**2)/(2*sigma**2))  # /(2*np.pi*sigma**2)


def laplacian_distribution(x, y, sigma): return (x**2+y**2)*gaussian_distribution(x, y, sigma)/(sigma**4) \
    - 2*gaussian_distribution(x, y, sigma)/(sigma**2)


def clip(image):
    image[image < 0] = 0
    image[image > 255] = 255
    return image.astype(int)


def brightness(image, factor):
    black = np.ones_like(image)
    image = (1-factor)*black+image*factor
    return clip(image)


def contrast(image, factor):
    gray = np.ones_like(image)*128
    image = (1-factor)*gray+factor*image
    return clip(image)


def get_kernel(size, sigma=1, type="g"):
    x, y = np.ogrid[-(size//2):size//2+1, -(size//2):size//2+1]
    if type == "g":
        kernel = gaussian_distribution(x, y, sigma)
        kernel = kernel/kernel.sum()
    elif type == "l":
        kernel = laplacian_distribution(x, y, sigma)
    return kernel


def gaussian(image, size=3, sigma=1):
    kernel = get_kernel(size, sigma, 'g')
    image_gaussian = convolve3D(image, kernel)
    return image_gaussian


def laplacian_of_gaussian(image, size=7, sigma=1):
    image = image/255
    kernel = get_kernel(size, sigma, 'l')
    image = convolve3D(image, kernel)
    image = image*255
    return image


def blur(image, sigma=.8):
    image = gaussian(image, 7, sigma)
    return clip(image)


def edges(image):
    image_log = laplacian_of_gaussian(image)
    image = clip(image_log)
    return image


def sharpen(image, factor=1):
    image_log = laplacian_of_gaussian(image)
    image_sharpen = image - factor*image_log
    return clip(image_sharpen)


def scale_x(image, sx):
    image_scaled = np.zeros(
        [int(sx*image.shape[0]), image.shape[1], 3], dtype=np.uint8)

    for i in range(image.shape[0]):
        image_scaled[int(i*sx), :, :] = image[i, :, :]

    return image_scaled


def scale_y(image, sy):
    image_scaled = np.zeros(
        [image.shape[0], int(sy*image.shape[1]), 3], dtype=np.uint8)

    for j in range(image.shape[1]):
        image_scaled[:, int(j*sy), :] = image[:, j, :]
    return image_scaled


def scale(image, sx, sy, interpolation_method="point"):
    image = scale_x(image, sx)
    image = scale_y(image, sy)

    if interpolation_method == 'point':
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9
        image = clip(convolve3D(image, kernel))

    elif interpolation_method == 'bilinear':
        kernel_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])/16
        kernel_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]])/8
        image[:, :, 0] = clip(convolve2D(image[:, :, 0], kernel_rb))
        image[:, :, 1] = clip(convolve2D(image[:, :, 1], kernel_rb))
        image[:, :, 2] = clip(convolve2D(image[:, :, 2], kernel_g))

    elif interpolation_method == 'gaussian':
        kernel = get_kernel(3, 1, 'g')
        image = clip(convolve3D(image, kernel))

    return image


def composition(back, fore, mask):
    mask_inv = np.bitwise_not(mask)
    fore_masked = np.bitwise_and(fore, mask)
    back_masked = np.bitwise_and(back, mask_inv)
    comp = fore_masked+back_masked
    return comp
