from sklearn.decomposition import NMF
import os
import numpy as np
from .segment_class import MitralSeg
from utils import thresholding_fn, window_detection
from evaluation import get_scores

dir_path = os.path.dirname(os.path.realpath(__file__))


class SegRNMF(MitralSeg):
    def __init__(self, sparsity_coef, max_iter, opt_flow_window_size,
                 search_window_size, threshold_wd, option='rnmf_seg',
                 window_size=(60, 80), init='nmf', rank=2, thresh1=95,
                 thresh2=98, stride=10, time_series_masking=False):

        super(SegRNMF, self).__init__()
        self.sparsity_coef = sparsity_coef
        self.max_iter = max_iter
        self.option = option
        self.k = rank
        self.init = init
        self.thresh1 = thresh1  # percentile for threshold
        self.thresh2 = thresh2
        self.window_size = window_size  # size of window for mitral valve
        self.stride = stride
        self.search_window_size = search_window_size
        self.opt_flow_window_size = opt_flow_window_size
        self.time_series_masking = time_series_masking
        self.threshold_wd = threshold_wd

    def rnmf(self, matrix2d, sparsity_coef):

        i = 0
        if self.init == 'nmf':
            model = NMF(n_components=self.k, init='random', random_state=0,
                        max_iter=800, tol=0.00001)
            W = model.fit_transform(matrix2d)
            H = model.components_
        else:
            W = np.random.uniform(0, 1, size=(self.n, self.k))
            H = np.random.uniform(0, 1, size=(self.k, self.m))

        while i <= self.max_iter:
            W_old = W
            H_old = H
            # initialize S matrix
            S = matrix2d - np.matmul(W_old, H_old)

            # update S matrix
            S[S > sparsity_coef / 2] = S[S > sparsity_coef / 2] - sparsity_coef / 2
            S[S < sparsity_coef / 2] = 0

            # update W matrix
            W_new = W_old * (np.matmul(np.maximum(matrix2d - S, 0), H_old.T)) / \
                    (np.matmul(np.matmul(W_old, H_old), H_old.T))
            nan_ind = np.isnan(W_new)
            inf_ind = np.isinf(W_new)
            W_new[nan_ind] = 0
            W_new[inf_ind] = 1
            W_new = W_new / np.linalg.norm(W_new, ord='fro', keepdims=True)

            # update H matrix
            H_new = H_old * (np.matmul(W_new.T, np.maximum(matrix2d - S, 0))) / \
                    (np.matmul(np.matmul(W_new.T, W_new), H_old))
            nan_ind = np.isnan(H_new)
            inf_ind = np.isinf(H_new)
            H_new[nan_ind] = 0
            H_new[inf_ind] = 1

            # normalize W and H
            W = W_new
            # H = H_new
            H = H_new * np.linalg.norm(W_new, ord='fro', keepdims=True)
            i += 1

        return W, H, S

    def train(self, save_location=None):

        def reshape_to_tensor(a): return np.reshape(a, newshape=(
        self.vert, self.horz, self.m))

        # RNMF on 2D representation
        print("RNMF #1")
        W1, H1, S1 = self.rnmf(self.matrix2d,
                               sparsity_coef=self.sparsity_coef[0])

        # threshold S
        S1 = thresholding_fn(S1, thresh=self.thresh1)

        # convert to tensor for window detection
        print("Convert to Tensor and Window Detection")
        W1H1 = np.matmul(W1, H1)
        W1H1 = reshape_to_tensor(W1H1)
        S1 = reshape_to_tensor(S1)

        win, _, _ = window_detection(tensor=S1, option=self.option,
                                     time_series=H1.T,
                                     window_size=self.window_size,
                                     search_window_size=self.search_window_size,
                                     opt_flow_window_size=self.opt_flow_window_size,
                                     stride=self.stride,
                                     time_series_masking=self.time_series_masking,
                                     threshold=self.threshold_wd)
        mask = win[0]

        # remove valve from RNMF reconstruction
        print("Removing Valve")
        M, M_prime = self.remove_valve(W1H1, S1, mask, self.thresh1)
        # reshape to 2D for RNMF on echo without valve to get muscle motion
        print("RNMF #2")
        M_prime = np.reshape(M_prime, (self.n, self.m))
        W, H, S = self.rnmf(M_prime, sparsity_coef=self.sparsity_coef[1])

        # get myocardium motion from RNMF by converting back to tensor (video)
        print("Getting Myocardium")
        W2H2 = np.matmul(W, H)
        myocardium = reshape_to_tensor(W2H2)
        # get valve by taking difference from original reconstruction and reconstruction without valve
        print("Getting Valve")
        valve = self.get_valve(M - myocardium, mask, self.thresh2,
                               morph_op=False, connected_struct=False)

        self.valve = valve
        self.myocardium = myocardium
        self.mask = mask
        self.noise = reshape_to_tensor(S)

        data_dict = {'valve': self.valve,
                     'myocardium': self.myocardium,
                     'mask': self.mask,
                     's': self.noise,
                     's_2': reshape_to_tensor(M - myocardium)}

        self.save_data(data_dict, save_location=save_location)
        eval_dict = get_scores(self.mask, self.valve, self.mask_gt,
                               self.valve_gt)
        return eval_dict
