import cv2
import numpy as np
from matplotlib import pyplot as plt
# mean square error formula


def MSE(block1, block2): return np.mean(np.subtract(
    block1.reshape(1, -1), block2.reshape(1, -1))**2)


def velocity_calculation(frame1, frame2, block_size, stride, max_step):
    velocity_vector = []

    for i in range(0, (frame1.shape[0]//block_size)*block_size, block_size):
        for j in range(0, (frame1.shape[1]//block_size)*block_size, block_size):
            block1 = frame1[i:i+block_size, j:j+block_size]

            # up search index
            num_stride_up = i//stride if i//stride < max_step else max_step
            start_x = i-num_stride_up*stride

            # down search index
            num_stride_down = (frame2.shape[0]-(i+block_size))//stride if (
                frame2.shape[0]-(i+block_size))//stride < max_step else max_step
            end_x = i+num_stride_down*stride

            # left search index
            num_stride_left = j//stride if j//stride < max_step else max_step
            start_y = j-num_stride_left*stride

            # right_serch_index
            num_stride_right = (frame2.shape[1]-(j+block_size))//stride if (
                frame2.shape[1]-(j+block_size))//stride < max_step else max_step
            end_y = j+num_stride_right*stride
            # list for mse and centroids of the matched block.
            mse_list, centroid_list = [], []
            for l in range(start_x, end_x, stride):
                for m in range(start_y, end_y, stride):
                    # block of the frame 2
                    block2 = frame2[l:l+block_size, m:m+block_size]
                    # mean square error
                    mse = MSE(block1, block2)
                    centroid_list.append((l+block_size//2, m+block_size//2))
                    mse_list.append(mse)

            min_index = mse_list.index(min(mse_list))
            # centroid point of the blocks of frame1
            p1 = (i+block_size//2, j+block_size//2)
            # centroid of the best matched patchs in frame2
            p2 = centroid_list[min_index]
            # velocity vector
            velocity_vector.append(np.subtract(p2, p1))

    return velocity_vector


def reconstruction(frame1, velocity, block_size):
    re_image = np.zeros_like(frame1)
    it = iter(velocity)
    for i in range(0, (frame1.shape[0]//block_size)*block_size, block_size):
        for j in range(0, (frame1.shape[1]//block_size)*block_size, block_size):
            dx, dy = next(it)
            re_image[i+dx:i+block_size+dx, j+dy:j+dy +
                     block_size] = frame1[i:i+block_size, j:j+block_size]

    return re_image
