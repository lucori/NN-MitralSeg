import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import os
from segment.nnmf import NNMF
from evaluation import get_scores
from .pytorch_utils import load_dataset, EarlyStopping
from .segment_class import MitralSeg
from sklearn.decomposition import NMF
from utils import window_detection
import numpy as np
from utils import animate, colorize, refactor, softplus
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from torch.utils.tensorboard import SummaryWriter
import time

dir_path = os.path.dirname(os.path.realpath(__file__))


class SegNNMF(MitralSeg):

    def __init__(self, l1_mult, l21_mult, embedding_mult, epochs, n_steps, learning_rate, mlp_size, gmf_size, batchsize,
                 num_workers, device, embedding_nmf_init, gmf_net_init, mlp_layers, threshold_layers, window_size,
                 save_data_every, save_tensorboard_summary_every, search_window_size, opt_flow_window_size,
                 train_test_split, patience, min_delta, early_stopping, connected_struct, morph_op, option,
                 threshold_mv, threshold_wd, spat_temp_mult):

        super(SegNNMF, self).__init__()
        self.mlp_size = mlp_size
        self.gmf_size = gmf_size
        self.l1_mult = l1_mult
        self.l21_mult = l21_mult
        self.embedding_mult = embedding_mult
        self.spat_temp_mult = spat_temp_mult
        self.epochs = epochs
        self.n_steps = n_steps
        self.lr = learning_rate
        self.batchsize = batchsize
        self.num_workers = num_workers
        self.device = device
        self.embedding_nmf_init = embedding_nmf_init
        self.gmf_net_init = gmf_net_init
        self.mlp_layers = mlp_layers
        self.threshold_layers = threshold_layers
        self.option = option
        self.nnmf = NNMF(self.gmf_size, self.mlp_size, self.mlp_layers, self.threshold_layers)
        self.window_size = window_size
        self.search_window_size = search_window_size
        self.opt_flow_window_size = float(opt_flow_window_size)
        self.connected_struct = connected_struct
        self.morph_op = morph_op
        self.train_test_split = train_test_split
        self.patience = patience
        self.min_delta = min_delta
        self.early_stopping = early_stopping
        self.save_data_every = save_data_every
        self.save_tensorboard_summary_every = save_tensorboard_summary_every
        self.x_hat = None
        self.s = None
        self.s_reshape = None
        self.train_loader = None
        self.val_loader = None
        self.embedding_params = None
        self.embedding_opt = None
        self.neu_mf_opt = None
        self.threshold_opt = None
        self.threshold_mv = threshold_mv
        self.threshold_wd = threshold_wd

    def l1_loss(self, s_out):
        loss = torch.mean(torch.abs(s_out))
        return loss

    def l21_loss(self, s):
        return torch.mean(torch.norm(s, dim=1) / np.sqrt(s.shape[1]))

    def set_x(self, matrix3d):
        super(SegNNMF, self).set_x(matrix3d)
        self.x_hat = torch.empty_like(torch.from_numpy(self.matrix2d), dtype=torch.float32)
        self.s = torch.empty_like(torch.from_numpy(self.matrix2d), dtype=torch.float32)
        embedding_nmf_init = self.initialize_embedding_gmf()
        self.nnmf.set_matrix(self.matrix2d, embedding_nmf_init)
        # create data loader
        print('loading dataset')
        if self.n_steps:
            self.epochs = int(self.n_steps / (matrix3d.size / self.batchsize))
        (self.train_loader, self.val_loader) = load_dataset(self.matrix2d, batch_size=self.batchsize,
                                                            num_workers=self.num_workers,
                                                            train_test_split=self.train_test_split)
        # optimizers for mlp and latent features
        self.embedding_params = self.nnmf.embedding_parameters()
        self.embedding_opt = optim.Adam(self.embedding_params, lr=self.lr)
        self.neu_mf_opt = optim.Adam(list(self.nnmf.mlp.parameters()) + list(self.nnmf.neu_mf.parameters()), lr=self.lr)
        self.threshold_opt = optim.Adam(self.nnmf.threshold_mlp.parameters(), lr=self.lr)

    def initialize_embedding_gmf(self):
        if self.embedding_nmf_init:
            model = NMF(n_components=self.gmf_size, init='random', random_state=0, max_iter=200, tol=0.0001)
            w = model.fit_transform(self.matrix2d)
            h = model.components_.transpose()
            nmf_par = (w, h)
        else:
            nmf_par = None
        return nmf_par

    def train_threshold(self, mse_xs_loss, l1_loss, l21_loss):
        self.threshold_opt.zero_grad()
        loss_threshold = mse_xs_loss + self.l1_mult * l1_loss + self.l21_mult * l21_loss
        loss_threshold.backward(retain_graph=True)
        self.threshold_opt.step()

    def train_neu_mf(self, mse_x_loss):
        self.neu_mf_opt.zero_grad()
        loss_neu_mf = mse_x_loss
        loss_neu_mf.backward(retain_graph=True)
        self.neu_mf_opt.step()

    def train_embedding(self, mse_x_loss, embedding_reg, spatial_ref, temporal_reg, valve=None):
        self.embedding_opt.zero_grad()
        embedding_loss = mse_x_loss + self.embedding_mult * embedding_reg + \
                         self.spat_temp_mult * (spatial_ref + temporal_reg)
        embedding_loss.backward(retain_graph=True)
        if valve:
            valve_frames = [int(list(v.keys())[0])-1 for v in valve]
            mid = int((valve_frames[-1]+valve_frames[-2])/2)
            for par in self.embedding_params:
                if par.shape[0] == self.m:
                    par.grad[mid:, ...] = 0
        self.embedding_opt.step()

    def train(self, save_location=None):
        # initialize epochs
        self.nnmf.init_params(gmf_net_init=self.gmf_net_init)
        self.nnmf.to(self.device)
        self.x_hat = self.x_hat.to(self.device)
        self.s = self.s.to(self.device)
        train_writer = SummaryWriter(log_dir=save_location + '/train')
        if self.val_loader:
            val_writer = SummaryWriter(log_dir=save_location + '/val')
        print('beginning training')
        ep = 0
        global_step = 0
        self.save_tensorboard_summary(train_writer, initialization=True)
        eval_dict = {}
        if self.early_stopping:
            self.early_stopping = EarlyStopping(patience=self.patience, mode='min', percentage=True,
                                            min_delta=self.min_delta)
        training_time = 0
        while ep < self.epochs:

            print("Epoch {} of {}".format(ep, self.epochs - 1))
            start_time_epoch = time.time()
            time_detach = 0
            self.nnmf.train()
            if self.early_stopping:
                cum_mse_xs_loss = 0
                cum_mse_x_loss = 0
                cum_l1_loss = 0
                cum_embedding_reg = 0
            for batch_id, batch in enumerate(self.train_loader, 0):
                pixel = Variable(batch[0])
                frame = Variable(batch[1])
                target = Variable(batch[2])

                # send x_hat and s to gpu
                pixel = pixel.to(self.device)
                frame = frame.to(self.device)
                target = target.to(self.device).float()
                target = torch.reshape(target, shape=(target.shape[0], 1))
                self.batchsize_eff = pixel.shape[0]
                # forward pass
                x_out, s_out = self.nnmf.forward(pixel, frame, target)
                self.s[pixel, frame] = torch.squeeze(s_out)
                # compute losses
                mse_xs_loss = nn.functional.mse_loss(target, x_out + s_out)
                mse_x_loss = nn.functional.mse_loss(target, x_out)
                l1_loss = self.l1_loss(s_out)
                l21_loss = 0  # self.l21_loss(self.s)
                embedding_reg = self.nnmf.embedding_regularization(pixel, frame)
                spatial_reg = self.nnmf.spatial_regularization(self.device)
                temporal_reg = self.nnmf.temporal_regularization(self.device)

                # backward and step
                self.train_neu_mf(mse_x_loss)

                self.train_embedding(mse_x_loss, embedding_reg, spatial_reg, temporal_reg)
                self.train_threshold(mse_xs_loss, l1_loss, l21_loss)

                # update x_hat and s
                start_time = time.time()
                self.x_hat[pixel, frame] = torch.squeeze(x_out.detach())
                self.s = self.s.detach()

                time_detach += time.time() - start_time
                training_time += time.time() - start_time_epoch

                time_detach = time_detach + time.time() - start_time

                if global_step % (self.epochs*5) == 0:
                    data_dict = {'mse_x': mse_x_loss,
                                 'mse_xs': mse_xs_loss,
                                 'embedding_regularization': embedding_reg,
                                 'l1_loss': l1_loss,
                                 'l21_loss': l21_loss}

                    # Print training progress every 100 steps
                    print(data_dict)

                    self.save_scalar_summary(data_dict, train_writer, global_step)
                    self.s_reshape = np.reshape(self.s.cpu().numpy(), newshape=(self.vert, self.horz, self.m))
                    self.myocardium = np.reshape(self.x_hat.cpu().numpy(), newshape=(self.vert, self.horz, self.m))

                if self.early_stopping:
                    cum_mse_xs_loss += mse_xs_loss.detach() / len(self.train_loader)
                    cum_mse_x_loss += mse_x_loss.detach() / len(self.train_loader)
                    cum_l1_loss += l1_loss.detach() / len(self.train_loader)
                    cum_embedding_reg += embedding_reg.detach() / len(self.train_loader)

                # batches
                global_step = global_step + 1

            if ep == 0 or (ep % self.save_data_every == 0 or ep == self.epochs - 1):
                print('extracting tensors for segmentation')
                start_time = time.time()
                self.s_reshape = np.reshape(self.s.cpu().numpy(), newshape=(self.vert, self.horz, self.m))
                self.myocardium = np.reshape(self.x_hat.cpu().numpy(), newshape=(self.vert, self.horz, self.m))
                print('finish extracting in ', time.time() - start_time, "seconds")
                print("window detection...")
                start_time = time.time()

                # detect window
                win, _, _ = window_detection(tensor=self.s_reshape, option=self.option,
                                             time_series=self.nnmf.gmf_v.weight.detach().cpu().numpy(),
                                             window_size=self.window_size,
                                             search_window_size=self.search_window_size,
                                             opt_flow_window_size=self.opt_flow_window_size,
                                             threshold=self.threshold_wd,
                                             stride=2)

                self.mask = win[0]

                print('finish window detection in ', time.time() - start_time, "seconds")
                start_time = time.time()
                self.valve = self.get_valve(self.s_reshape, self.mask, threshold=self.threshold_mv)
                print('finish valve segmentation in ', time.time() - start_time, "seconds")

                # saving segmentation and predicted window as well the embedding and sparse matrix
                data_dict_save = self.create_dict(ep)
                if ep != 0:
                    self.save_data(data_dict_save, save_location=save_location)

                # get evaluation scores
                start_time = time.time()
                if self.valve_gt is not None and self.mask_gt is not None:
                    eval_dict = get_scores(self.mask, self.valve, self.mask_gt, self.valve_gt)
                    self.save_scalar_summary(eval_dict, train_writer, global_step)
                    print('finish scalar eval summary in ', time.time() - start_time, "seconds")

            if self.early_stopping:
                if self.early_stopping.step(cum_mse_x_loss):
                    self.save_tensorboard_summary(train_writer, initialization=False, global_step=global_step)
                    break

            start_time = time.time()
            if ep % self.save_tensorboard_summary_every == 0 or ep == self.epochs - 1:
                self.save_tensorboard_summary(train_writer, initialization=False, global_step=global_step)
                pass
            print('finish image summary in ', time.time() - start_time, "seconds")

            if self.val_loader:
                with torch.no_grad():
                    self.nnmf.eval()
                    mse_xs_loss = 0
                    mse_x_loss = 0
                    l1_loss = 0
                    embedding_reg = 0
                    spatial_reg = 0
                    temporal_reg = 0
                    for batch_id, batch in enumerate(self.val_loader, 0):
                        pixel, frame, target = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
                        # send x_hat and s to gpu
                        pixel = pixel.to(self.device)
                        frame = frame.to(self.device)
                        target = target.to(self.device).float()
                        target = torch.reshape(target, shape=(target.shape[0], 1))
                        self.batchsize_eff = pixel.shape[0]
                        # forward pass
                        x_out, s_out = self.nnmf.forward(pixel, frame, target)

                        # compute losses
                        mse_xs_loss += nn.functional.mse_loss(target, x_out + s_out)
                        mse_x_loss += nn.functional.mse_loss(target, x_out)
                        l1_loss += self.l1_loss(s_out)
                        embedding_reg += self.nnmf.embedding_regularization(pixel,
                                                                            frame) * self.embedding_mult / self.batchsize_eff
                        spatial_reg += self.nnmf.spatial_regularization(self.device)
                        temporal_reg += self.nnmf.temporal_regularization(self.device)

                    mse_xs_loss = mse_xs_loss / len(self.val_loader)
                    mse_x_loss = mse_x_loss / len(self.val_loader)
                    l1_loss = l1_loss / len(self.val_loader)
                    embedding_reg = embedding_reg / len(self.val_loader)
                    spatial_reg = spatial_reg / len(self.val_loader)
                    temporal_reg = temporal_reg / len(self.val_loader)
                    data_dict = {'mse_x': mse_x_loss,
                                 'mse_xs': mse_xs_loss,
                                 'embedding_regularization': embedding_reg,
                                 'l1_loss': l1_loss,
                                 'spatial_reg': spatial_reg,
                                 'temporal_reg': temporal_reg}

                    self.save_scalar_summary(data_dict, val_writer, global_step)
            ep = ep + 1

        eval_dict.update({'time': training_time})
        return eval_dict

    def save_scalar_summary(self, dict, writer, global_step):
        for name, loss in dict.items():
            writer.add_scalar(name, loss, global_step=global_step)

    def save_plots(self, data_dict, ep):
        animate(self.matrix3d, data_dict['valve.npy'], self.dir + 'valve' + str(ep) + '.mp4')
        animate(self.matrix3d, data_dict['myocardium.npy'], self.dir + 'myocardium' + str(ep) + '.mp4')
        animate(self.matrix3d, data_dict['s.npy'], self.dir + 's' + str(ep) + '.mp4')

        fig, ax = plt.subplots(figsize=(8, 8), nrows=2, ncols=2)
        U = data_dict['U'].reshape((self.vert, self.horz, self.d))
        ax[0, 0].imshow(U[:, :, 0], cmap='binary')
        ax[0, 1].imshow(U[:, :, 1], cmap='binary')
        plt.savefig(self.dir + 'U' + str(ep) + '.jpg')

        plt.plot(data_dict['V'])
        plt.savefig(self.dir + 'V' + str(ep) + '.jpg')

    def save_tensorboard_embeddings(self, u, v, embedding_dim, name_u, name_v, writer, global_step, matrix_bin):
        u = softplus(u.weight.detach().cpu().numpy().reshape((self.vert, self.horz, embedding_dim)))
        u = np.expand_dims(np.stack([u, u, u], axis=0), axis=0)
        writer.add_images(name_u, refactor(u), global_step=global_step)
        v = softplus(v.weight.detach().cpu().numpy())
        fig = plt.figure()
        plt.plot(v)
        writer.add_figure(name_v, fig, global_step=global_step)
        dot_product = np.dot(u, v.T)[0, 0, ...]
        myocardium_dot_prod = self.get_video(dot_product, matrix_bin, cmap='rainbow')
        writer.add_video(name_u + '_' + name_v + '_dotprod', myocardium_dot_prod, global_step=global_step)

    def save_tensorboard_summary(self, writer, initialization=True, global_step=0):
        matrix = np.transpose(self.matrix3d, axes=(2, 0, 1))
        matrix_bin = colorize(matrix, cmap='binary')

        for j, l in enumerate(self.nnmf.neu_mf.parameters()):
            writer.add_histogram('weights_neu_mf' + str(j), l.data, global_step=global_step)


        inp = np.expand_dims(np.linspace(-1, 1), axis=1)
        out = self.nnmf.threshold_mlp.forward(torch.from_numpy(inp).to(self.device).float()).detach().cpu().numpy()
        fig = plt.figure()
        plt.plot(inp, out)
        writer.add_figure('threshold_fun', fig, global_step=global_step)

        if self.mlp_size > 0:
            #for j, l in enumerate(self.nnmf.mlp.parameters()):
            #    writer.add_histogram('weights_' + str(j), l.data, global_step=global_step)
            self.save_tensorboard_embeddings(self.nnmf.mlp_u, self.nnmf.mlp_v, self.mlp_size, 'mlp_u', 'mlp_v', writer,
                                             global_step, matrix_bin)
        if self.gmf_size > 0:
            self.save_tensorboard_embeddings(self.nnmf.gmf_u, self.nnmf.gmf_v, self.gmf_size, 'gmf_u', 'gmf_v', writer,
                                             global_step, matrix_bin)
        # for first time
        if not initialization:
            myocardium = self.get_video(self.myocardium, matrix_bin,  cmap='rainbow')
            noise = self.get_video(self.s_reshape, matrix_bin, cmap='rainbow')
            myocardium = self.get_video(self.myocardium, matrix_bin, cmap='rainbow')
            sparse = self.get_video(self.s_reshape, matrix_bin, cmap='rainbow')
            writer.add_video('myocardium', myocardium, global_step=global_step)
            writer.add_video('sparse', sparse, global_step=global_step)
            valve = self.get_video(self.valve, matrix_bin, cmap='rainbow')
            writer.add_video('valve', valve, global_step=global_step)

        # predicted and ground truth valves
        if self.valve_gt is not None:
            fig, axs = plt.subplots(1, 3)
            fig.set_size_inches(12, 4)
            fig.suptitle('')
            for i in range(len(self.valve_gt)):
                axs[i].imshow(self.get_valve_image(i, initialization))
            writer.add_figure('segmentation', fig, global_step=global_step)

        # predicted and ground truth window
        if self.mask_gt is not None:
            fig = plt.figure()
            frame = np.squeeze(self.matrix3d[..., 0])

            if initialization:
                mask = np.zeros(shape=self.mask_gt.shape)
            else:
                if len(self.mask.shape) == 3:
                    mask = np.squeeze(self.mask[..., 0])
                else:
                    mask = self.mask

            color_image = np.clip(np.dstack([0.75 * frame + mask, 0.75 * frame, 0.75 * frame + self.mask_gt]), a_min=0,
                                  a_max=1)
            plt.imshow(color_image)
            writer.add_figure('window', fig, global_step=global_step)

    def get_video(self, tensor, matrix_bin, cmap='binary'):
        tensor = np.transpose(tensor, axes=(2, 0, 1))
        tensor_col = colorize(tensor, cmap=cmap)
        tensor = np.where(np.stack([tensor for _ in range(4)], axis=-1), tensor_col, matrix_bin)
        tensor = np.transpose(tensor, axes=(0, 3, 1, 2))
        tensor = np.expand_dims(tensor, axis=0)
        return tensor

    def create_dict(self, ep):
        ep = str(ep)
        get_name = lambda x: x + ep + '.npy'
        data_dict = {get_name('valve'): self.valve,
                     get_name('myocardium'): self.myocardium,
                     get_name('mask'): self.mask,
                     get_name('s'): self.s_reshape}
        if self.mlp_size != 0:
            data_dict.update({get_name('mlp_u'): self.nnmf.mlp_u.weight.detach().cpu().numpy(),
                              get_name('mlp_v'): self.nnmf.mlp_v.weight.detach().cpu().numpy()})
        if self.gmf_size != 0:
            data_dict.update({get_name('gmf_u'): self.nnmf.gmf_u.weight.detach().cpu().numpy(),
                              get_name('gmf_v'): self.nnmf.gmf_v.weight.detach().cpu().numpy()})

        inp = np.expand_dims(np.linspace(-1, 1), axis=1)
        out = self.nnmf.threshold_mlp.forward(torch.from_numpy(inp).to(self.device).float()).detach().cpu().numpy()
        data_dict.update({'threshold_fun' + ep + '.npy': out})
        return data_dict

    def get_valve_image(self, idx, initialization):

        valve_idx = int(list(self.valve_gt[idx].keys())[0]) - 1
        valve_values = list(self.valve_gt[idx].values())[0]
        frame = np.squeeze(self.matrix3d[..., valve_idx])

        if initialization:
            valve_pred = np.zeros(shape=valve_values.shape)
        else:
            valve_pred = np.squeeze(self.valve[..., valve_idx])

        valve_image = np.clip(np.dstack([0.75 * frame + valve_pred,
                                         0.75 * frame,
                                         0.75 * frame + valve_values]), a_min=0, a_max=1)

        return valve_image
