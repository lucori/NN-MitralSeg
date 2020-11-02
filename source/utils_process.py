import cv2
import os
import imageio
import numpy as np

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def resize_frame(frame, new_side_len):
    #   min_height_chamber2 = 506
    #   min_width_chamber2 = 416
    #
    #   max_height_chamber2 = 708
    #   max_width_chamber2 = 1036
    #
    #   min_height_chamber4 = 514
    #   min_width_chamber4 = 446
    #
    #   max_height_chamber4 = 718
    #   max_width_chamber4 = 1104

    height = frame.shape[0]
    width = frame.shape[1]

    max_side_len = max(height, width)

    # padding with np.nan values in order to get a square input.
    if max_side_len == height:

        frame_pad = np.empty((height, height))
        frame_pad[:] = np.nan
        padding = max_side_len - width

        if (padding % 2) == 0:
            frame_pad[:, range((padding // 2), (max_side_len - padding // 2))] = frame
        else:
            frame_pad[:, range(((padding + 1) // 2), (max_side_len - (padding - 1) // 2))] = frame

    else:

        frame_pad = np.empty((width, width))
        frame_pad[:] = np.nan
        padding = max_side_len - height

        if (padding % 2) == 0:
            frame_pad[range((padding // 2), (max_side_len - padding // 2)), :] = frame
        else:
            frame_pad[range(((padding + 1) // 2), (max_side_len - (padding - 1) // 2)), :] = frame

    # resizing
    frame_res = cv2.resize(frame_pad, (new_side_len, new_side_len), interpolation=cv2.INTER_AREA)
    return frame_res


def crop_outer_part(img, info):
    """ remove parts around the ECHO """

    lower = int(info["height"] / 10)  # remove lower part because of changing time in the bar
    upper = int(info["height"] / 8)  # 5.8)  # remove upper part because of the moving ECG
    right = int(info["width"] / 7.7)  # remove right part because of changing pulse (ppm) display

    img[0:lower:, :] = 0
    img[-upper:, :] = 0
    img[:, -right:] = 0

    return img


def get_slope(x1, y1, x2, y2):
    # returns the slope of a line going between the two points
    try:
        return (float(y2) - y1) / (float(x2) - x1)
    except ZeroDivisionError:
        # line is vertical
        return None


def yintercept(x, y, slope):
    """Get the y intercept of a line segment"""
    if slope is not None:
        return y - slope * x
    else:
        return None


def get_corner_point(mask, corner_specification):
    """ calculates the corner points of the mask """

    height = mask.shape[0]
    width = mask.shape[1]

    if corner_specification == "lowest":
        # start at the top left
        for x in range(0, height):
            for y in range(0, width):
                if mask[x, y] == 1:
                    return x, y

    if corner_specification == "highest":
        # start from the bottom left
        for x in range(height - 1, 0, -1):
            for y in range(0, width):
                if mask[x, y] == 1:
                    return x, y

    if corner_specification == "right":
        # start from the bottom right
        for y in range(width - 1, 0, -1):
            for x in range(height - 1, 0, -1):
                if mask[x, y] == 1:
                    return x, y

    if corner_specification == "left":
        # start from the bottom left
        for y in range(0, width):
            for x in range(height - 1, 0, -1):
                if mask[x, y] == 1:
                    return x, y


def remove_colored_pixel(frame_bgr):
    """removes colored pixels and returns a grey scale image """

    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # set all colored pixels to black
    frame_hsv[frame_hsv[:, :, 1] > 80] = 0

    # convert from HSV to BRG to GREY
    frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return frame_grey


def morphological_transformation_foreground(foreground): 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    vertical = int(foreground.shape[0] / 5)
    horizontal = int(foreground.shape[1] / 5)
    foreground[0:vertical, 0:horizontal] = 0
    foreground[0:vertical, -horizontal:] = 0

    return foreground


def morphological_transformation_mask(mask): 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def standardize_ndarray(ndarray):

    # takes a n-dim array and standardizes it to zero mean and std 1 ignoring 'nan' values.
    # takes an ECHO and standardizes the video to zero mean and std. of one.
    # the result can optionally be rescaled to greyscale (values from 0 to 255)

    mean = np.nanmean(ndarray, keepdims=True)
    std = np.nanstd(ndarray, keepdims=True)
    z = np.divide((ndarray - mean), std)

    return z


def rgb_2_gray(frame):

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def calculate_gradients(array_2d):
    laplacian = cv2.Laplacian(array_2d, cv2.CV_64F)
    sobelx = cv2.Sobel(array_2d, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(array_2d, cv2.CV_64F, 0, 1, ksize=5)

    return laplacian


def opt_triangular_points(mask,foreground,validate_cropping,out_file_dir,file_path,file_name):

    height = foreground.shape[0]
    width = foreground.shape[1]
    x_low, y_low = get_corner_point(foreground, "lowest")
    x_high, y_high = get_corner_point(foreground, "highest")
    x_right, y_right = get_corner_point(foreground, "right")
    x_left, y_left = get_corner_point(foreground, "left")

    if np.sum(mask) > mask.shape[0] * mask.shape[1] * 0.9:
        print("Broken (all white) overlay!")
    else:
        # correct lowest point down to triangle peak
        y_1 = y_low
        while mask[x_low, y_1] > 0:
            # do not go further left when in the "writing" bar
            if np.sum(mask[x_low, :]) > width / 2:
                break
            y_1 -= 1

        y_2 = y_low
        while mask[x_low, y_2] > 0:
            # do not go further right when in the "writing" bar
            if np.sum(mask[x_low, :]) > width / 2:
                break
            y_2 += 1

        # take middle point and go down
        y_low_opt = int((y_1 + y_2) / 2)

        x_low_opt = x_low
        while x_low_opt > 0 and mask[x_low_opt, y_low_opt] > 0:
            # do not go further down when the "writing" bar is reached
            if np.sum(mask[x_low_opt, :]) > width / 2:
                x_low_opt += 5  # correct for some pixel
                break
            x_low_opt -= 1

        # correct left and right points
        y_left_opt = y_left
        while y_left_opt > 0 and mask[x_left, y_left_opt] > 0:
            y_left_opt -= 1

        y_right_opt = y_right
        while y_right_opt < width and mask[x_right, y_right_opt] > 0:
            y_right_opt += 1

        # use optimized points
        x_low = x_low_opt
        y_low = y_low_opt

        y_left = y_left_opt
        y_right = y_right_opt

    # first line
    m1 = get_slope(x_low, y_low, x_left, y_left)
    b1 = yintercept(x_low, y_low, m1)

    # second line
    m2 = get_slope(x_low, y_low, x_right, y_right)
    b2 = yintercept(x_low, y_low, m2)

    # optimize (second step)
    x_middle = int((x_low + x_high) / 2)
    y_middle_left = np.math.ceil((m1 * x_middle + b1))
    y_middle_right = int(m2 * x_middle + b2)

    # correct left and right points
    y_middle_left_opt = y_middle_left
    while y_middle_left_opt > 0 and mask[x_middle, y_middle_left_opt] > 0:
        y_middle_left_opt -= 1

    y_middle_right_opt = y_middle_right
    while y_middle_right_opt < width and mask[x_middle, y_middle_right_opt] > 0:
        y_middle_right_opt += 1

    if y_middle_right_opt >= width:
        y_middle_right_opt = width - 1

        # right opt. line
        m2 = get_slope(x_low, y_low, x_high, y_middle_right_opt)
        b2 = yintercept(x_low, y_low, m2)

    else:

        # right opt. line
        m2 = get_slope(x_low, y_low, x_middle, y_middle_right_opt)
        b2 = yintercept(x_low, y_low, m2)

    # left opt. line
    m1 = get_slope(x_low, y_low, x_middle, y_middle_left_opt)
    b1 = yintercept(x_low, y_low, m1)

    # correct left/right (y-coord) corner points
    if y_right < y_middle_right_opt:
        y_right = y_middle_right_opt
        x_right = x_middle

    if y_left > y_middle_left_opt:
        y_left = y_middle_left_opt
        x_left = x_middle

    for x in range(0, height):
        for y in range(0, width):

            # select all points inside the triangle
            if m1 * x + b1 <= y <= m2 * x + b2 and x < x_high:
                foreground[x, y] = 1
            else:
                foreground[x, y] = np.nan

    return foreground, (x_low, y_left), (x_high, y_right)


def save_picture(out_file_dir, file_path, pic):
    out_file_path = os.path.join(out_file_dir, os.path.splitext(os.path.basename(file_path))[0] + '.jpg')
    os.makedirs(out_file_dir, exist_ok=True)
    imageio.imwrite(out_file_path, pic)