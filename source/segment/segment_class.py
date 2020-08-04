import os
import numpy as np
import medpy.filter.smoothing as mp
from utils import tensor_to_matrix
from utils import thresholding_fn
from scipy.ndimage.measurements import label
import cv2
dir_path = os.path.dirname(os.path.realpath(__file__))


class MitralSeg:

    def __init__(self):
        self.save_location = None
        self.valve = None
        self.valve_gt = None
        self.myocardium = None
        self.mask = None
        self.mask_gt = None
        self.matrix3d = None
        self.vert = None
        self.horz = None
        self.m = None
        self.n = None
        self.matrix2d = None
        self.dir = None

    def set_x(self, matrix3d):
        self.matrix3d = matrix3d / 255
        self.vert, self.horz, self.m = matrix3d.shape
        self.n = self.horz * self.vert
        self.matrix2d = tensor_to_matrix(self.matrix3d, self.n, self.m)

    def train(self):
        pass

    def set_save_location(self, save_location):
        self.save_location = save_location
        self.dir = dir_path + '/../../out/' + self.__class__.__name__
        if self.option == 'optical_flow':
            self.dir += '/optical_flow/' + self.save_location + '/'
        else:
            self.dir += '/original/' + self.save_location + '/'

    def set_labels(self, labels):
        self.mask_gt = labels['box']
        self.valve_gt = labels['masks']

    def save_data(self, data_dict, save_location=None):
        if save_location:
            self.set_save_location(save_location)
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        for k in data_dict.keys():
            np.save(file=self.dir + k, arr=data_dict[k])

    @staticmethod
    def remove_valve(WH, S, mask, threshold):
        M = WH + S
        # take window of threshold S matrix
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, -1)
        S_prime = mask * thresholding_fn(S, thresh=threshold)
        # anisotropic diffusion to connect segments
        S_aniso = np.empty_like(S_prime)
        for j in range(S_prime.shape[2]):
            S_aniso[:, :, j] = mp.anisotropic_diffusion(S_prime[:, :, j], niter=5, kappa=20)

        # remove from rnmf
        S_aniso = np.reshape(S_prime, newshape=M.shape)
        M_prime = M - M * S_aniso
        return M, M_prime

    @staticmethod
    def get_valve(sparse, mask, threshold, thresh_func='percentile', morph_op=True, connected_struct=True):

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        if mask.shape[2] == 1:
            mask = np.repeat(mask, sparse.shape[-1], axis=-1)

        thresh = thresholding_fn(sparse, mask=mask, thresh=threshold, thresh_func=thresh_func)
        valve_masked = thresh * mask
        valve_aniso = np.empty_like(valve_masked)

        kernel_erode = np.ones((2, 2), np.uint8)
        kernel_dilated = np.ones((1, 1), np.uint8)
        # anisotropic diffusion to connect segments1, gamma=10,
        for j in range(sparse.shape[2]):

            valve_diffused = mp.anisotropic_diffusion(valve_masked[:, :, j], niter=2, kappa=20, option=3)

            if morph_op:
                valve_diffused = cv2.erode(valve_diffused, kernel_erode, iterations=1)
                valve_diffused = cv2.dilate(valve_diffused, kernel_dilated, iterations=1)

            valve_aniso[:, :, j] = valve_diffused

        valve_aniso[valve_aniso > 0] = 1

        if connected_struct:
            structure = np.ones((3, 3, 3), dtype=np.int)

            # find connected commponents
            labeled, ncomp = label(valve_aniso, structure)
            components = sorted([(n, np.sum(labeled[labeled == n])) for n in range(1, ncomp + 1)],
                                key=lambda x: x[1],
                                reverse=True)

            print("Components: ", components)

            if len(components) == 0:
                print("No components detected")
                return valve_aniso

            valve_aniso = np.zeros(labeled.shape)
            valve_aniso[labeled == components[0][0]] = 1

            # include other large components as well
            if len(components) > 1:
                c_idx = 1
                while components[c_idx][1] > 0.3 * components[0][1]:
                    print("Including next largest connected component.")
                    valve_aniso[labeled == components[c_idx][0]] = 1
                    c_idx += 1

                    if len(components) == c_idx:
                        print("Trying to include non-existing component.")
                        break

        return valve_aniso