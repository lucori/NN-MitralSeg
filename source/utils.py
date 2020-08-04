import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import cv2
import numpy as np
import os
from numpy.ma import masked_array


def animate(tensor, valve=None, name=None):
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(j):
        ax.imshow(masked_array(tensor[:, :, j], 1 - valve[:, :, j]),
                  cmap='rainbow')
        ax.imshow(-masked_array(tensor[:, :, j], valve[:, :, j]),
                  cmap='binary')
        ax.set_axis_off()

    anim = FuncAnimation(fig, update, frames=tensor.shape[-1], interval=100)
    writer = animation.writers['ffmpeg'](fps=30)

    anim.save(name, writer=writer, dpi=100)
    plt.close(fig)
    return anim


def optical_flow(tensor, winsize):
    if winsize <= 1:
        winsize = int(winsize * tensor.shape[1])

    flow = np.zeros(
        shape=(tensor.shape[0], tensor.shape[1], tensor.shape[2] - 1))
    for i in range(tensor.shape[2] - 1):
        flow_val = cv2.calcOpticalFlowFarneback(tensor[:, :, i],
                                                tensor[:, :, i + 1], flow=None,
                                                pyr_scale=0.5,
                                                levels=1,
                                                winsize=winsize, iterations=3,
                                                poly_n=7, poly_sigma=3.5,
                                                flags=1)
        flow[:, :, i] = np.sqrt(
            np.square(flow_val[:, :, 1]) + np.square(flow_val[:, :, 0]))

    return flow


def thresholding_fn(tensor, thresh, mask=None, thresh_func='percentile'):
    matrix = np.copy(
        tensor)  # copy to prevent that the passed tensor is modified

    if thresh_func == 'percentile':
        # calculate threshold value for given percentile
        if mask is not None:
            # calculate treshhold based on values inside window!
            thresh_val = np.percentile(matrix*mask, q=thresh)
        else:
            thresh_val = np.percentile(matrix, q=thresh)
        # assign binary values to pixels above and below threshold
        matrix[matrix < thresh_val] = matrix[matrix < thresh_val] * 0

    else:
        for i in range(matrix.shape[2]):
            matrix[:, :, i] = cv2.GaussianBlur(
                np.array(matrix[:, :, i] * 255, dtype=np.uint8), (21, 21),
                sigmaX=0.5, sigmaY=0.5)
            _, matrix[:, :, i] = cv2.threshold(
                np.array(matrix[:, :, i], dtype=np.uint8), 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return matrix


def denoise(h, matrix):
    denoised = np.empty_like(matrix)

    for i in range(matrix.shape[2]):
        denoised[:, :, i] = cv2.fastNlMeansDenoising(
            np.array(matrix[:, :, i] * 255, dtype=np.uint8), h=h,
            templateWindowSize=5, searchWindowSize=11)
    denoised = denoised / 255
    return denoised


def find_window(tensor, win_size, search_win_size, stride=1,
                offset_fac_vert=0.0, offset_fac_horz=0.0):
    # works as well for integrated (over time) input tensor
    if len(tensor.shape) == 2:
        tensor = np.expand_dims(tensor, axis=-1)

    vert = tensor.shape[0]
    horz = tensor.shape[1]

    if win_size[0] <= 1:
        win_size_vert = int(vert * win_size[0])
        win_size_horz = int(horz * win_size[1])
    else:
        win_size_vert = int(win_size[0])
        win_size_horz = int(win_size[1])

    # important tuning parameter
    if search_win_size[0] <= 1:
        search_win_size_vert = int(vert * search_win_size[0])
        search_win_size_horz = int(horz * search_win_size[1])
    else:
        search_win_size_vert = int(search_win_size[0])
        search_win_size_horz = int(search_win_size[1])

    norms = np.zeros(
        shape=(vert - search_win_size_vert, horz - search_win_size_horz))

    for i in np.arange(0, vert - search_win_size_vert, stride):
        for j in np.arange(0, horz - search_win_size_horz, stride):
            norms[int(i), int(j)] = np.linalg.norm(
                tensor[int(i):int(i) + search_win_size_vert,
                int(j):int(j) + search_win_size_horz,
                :])

    # get index for window with maximum Frobenius norm
    window_index = np.unravel_index(np.argmax(norms), shape=norms.shape)

    # create mask for window index
    mask = np.zeros(shape=(vert, horz))
    hgt_ind, wdh_ind = window_index

    offset_vert = int(offset_fac_vert * win_size_vert)  # engineered
    offset_horz = int(offset_fac_horz * win_size_horz)  # engineered

    hgt_ind += int(search_win_size_vert / 2) + offset_vert
    wdh_ind += int(search_win_size_horz / 2) + offset_horz

    mask[np.max([0, hgt_ind - int(win_size_vert / 2)]):(
                hgt_ind + int(win_size_vert / 2)),
    np.max([0, wdh_ind - int(win_size_horz / 2)]):(
                wdh_ind + int(win_size_horz / 2))] = 1

    return mask


def get_mask(s, option, win_size, s_win_size, opt_flow_window_size,
             stride, thresh, time_series=None, t_idx=None):
    # mask by time series
    t = time_series
    if t_idx is not None:

        t_ = t[..., t_idx]
        # resacle to 0, 1
        t_ = (t_ - np.min(t_)) / (np.max(t_) - np.min(t_))
        s_ = np.moveaxis(
            np.asarray([s[..., i] * t_[i] for i in range(t.shape[0])]), 0, -1)
    else:
        s_ = s

    # Threshold sparse matrix
    s_thresh = thresholding_fn(s_, thresh=thresh)

    if option == 'optical_flow':
        if opt_flow_window_size <= 1:
            opt_flow_window_size = int(opt_flow_window_size * s.shape[0])
        else:
            opt_flow_window_size = int(opt_flow_window_size)

        s_thresh = optical_flow(s_thresh, winsize=opt_flow_window_size)

    assert len(s_thresh.shape) == 3

    # find window
    mask_thresh = find_window(s_thresh, win_size, s_win_size, stride=stride,
                              offset_fac_vert=0.05, offset_fac_horz=0.05)

    return mask_thresh, np.sum(np.abs(s_thresh), axis=-1)


def window_detection(tensor, option, window_size, search_window_size,
                     opt_flow_window_size, stride, threshold,
                     time_series=None, time_series_masking=True):
    # apply softplus to the time series
    if time_series_masking and time_series is not None:
        time_series = softplus(time_series)
    else:
        time_series_masking = False

    vert = tensor.shape[0]
    horz = tensor.shape[1]

    if window_size[0] <= 1:
        win_size_vert = int(vert * window_size[0])
        win_size_horz = int(horz * window_size[1])
    else:
        win_size_vert = int(window_size[0])
        win_size_horz = int(window_size[1])

    mask, opt_flow = get_mask(tensor, option, window_size, search_window_size,
                              opt_flow_window_size,
                              time_series=time_series, t_idx=None,
                              thresh=threshold,
                              stride=stride)

    if time_series_masking:
        mask_0, opt_flow_0 = get_mask(tensor, option, window_size,
                                      search_window_size, opt_flow_window_size,
                                      time_series=time_series, t_idx=0,
                                      thresh=threshold, stride=stride)
        mask_1, opt_flow_1 = get_mask(tensor, option, window_size,
                                      search_window_size, opt_flow_window_size,
                                      time_series=time_series, t_idx=1,
                                      thresh=threshold, stride=stride)

        mid_point_vert_0 = int(
            (np.where(mask_0)[0][0] + np.where(mask_0)[0][-1]) / 2)
        mid_point_vert_1 = int(
            (np.where(mask_1)[0][0] + np.where(mask_1)[0][-1]) / 2)

        mid_point_horz_0 = int(
            (np.where(mask_0)[1][0] + np.where(mask_0)[1][-1]) / 2)
        mid_point_horz_1 = int(
            (np.where(mask_1)[1][0] + np.where(mask_1)[1][-1]) / 2)

        # check if overlap
        allowed_overlap = 0.5
        if np.sum(np.logical_and(mask_0, mask_1)) < allowed_overlap * np.sum(
                mask_1):
            print("Two different valves detected! Take right one!")

            # take right window
            if mid_point_horz_0 > mid_point_horz_1:
                mask = mask_0
                opt_flow = opt_flow_0
            else:
                mask = mask_1
                opt_flow = opt_flow_1

        else:
            print("Two windows do partially overlap. Take mid-point!")

            mask = np.zeros(mask_0.shape)
            mid_vert = int((mid_point_vert_0 + mid_point_vert_1) / 2)
            mid_horz = int((mid_point_horz_0 + mid_point_horz_1) / 2)

            vert_off = int(win_size_vert / 2)
            horz_off = int(win_size_horz / 2)

            mask[np.max([0, mid_vert - vert_off]):mid_vert + vert_off,
            np.max([mid_horz - horz_off]):mid_horz + horz_off] = 1.0

        return (mask, opt_flow), (mask_0, opt_flow_0), (mask_1, opt_flow_1)

    return (mask, opt_flow), (mask, opt_flow), (mask, opt_flow)


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in
                        open('tmp', 'r').readlines()]
    return np.argmax(memory_available)


def tensor_to_matrix(matrix3d, n, m):
    # bring time dimension to front and flatten
    matrix2d = np.reshape(matrix3d, (n, m))
    return matrix2d


def matrix_to_pixel_frame_target(matrix):
    target_vales = matrix.flatten()
    pixel_ind = np.repeat(np.arange(matrix.shape[0]), matrix.shape[1])
    frame_ind = np.tile(np.arange(matrix.shape[1]), matrix.shape[0])
    index_mat = np.vstack((pixel_ind, frame_ind, target_vales)).T
    return index_mat


def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)

    Returns a 4D uint8 tensor of shape [height, width, 4].
    """
    import matplotlib
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    value = value.squeeze()

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    return value


def refactor(im_in):
    shape = im_in.shape
    im_list = [im_in[..., i] for i in range(shape[-1])]
    im_out = np.concatenate(im_list, axis=3)
    return im_out


def get_valve_image(idx, dt, pred):
    labels = dt.labels
    matrix = np.nan_to_num(dt.matrix3d) / 255.0

    valve_idx = int(list(labels['masks'][idx].keys())[0]) - 1
    valve_values = list(labels['masks'][idx].values())[0]
    frame = np.squeeze(matrix[..., valve_idx])

    valve_pred = np.squeeze(pred[..., valve_idx])
    valve_image = np.clip(np.dstack([0.75 * frame + valve_pred,
                                     0.75 * frame,
                                     0.75 * frame + valve_values]), a_min=0,
                          a_max=1)

    return valve_image


def softplus(x):
    return np.where(x == -np.infty, 0,
                    x * (x >= 0) + np.log1p(np.exp(-np.abs(x))))


def softminus(x):
    return np.where(x == 0, -np.infty, x + np.log1p(-np.exp(-x)))
