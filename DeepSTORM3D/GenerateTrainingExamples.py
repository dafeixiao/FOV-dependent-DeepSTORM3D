# Import modules and libraries
import torch
import numpy as np
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from DeepSTORM3D.data_utils import generate_batch, complex_to_tensor
from DeepSTORM3D.physics_utils import calc_bfp_grids, EmittersToPhases, PhysicalLayer, NoiseLayer
from DeepSTORM3D.helper_utils import CalcMeanStd_All
import os
import pickle
from PIL import Image
import argparse
import torch
from helper_utils_ import show_imset, noise_processing
import scipy.io as sio
import numpy as np
import glob
from PSFI3D.PSF_interpolation import FDPSFGenerator0
import pickle
from skimage import io
from torch import nn
import scipy


class TrainingImageLayer(nn.Module):
    def __init__(self, setup_params):
        super().__init__()
        self.psf_module = setup_params['psf_generator']  # interpolation
        self.noise_layer = NoiseLayer(setup_params)  # add noise
        self.interpolated_psf_width = setup_params['interpolated_psf_width']
        margin = np.ceil(setup_params['interpolated_psf_width']/2-setup_params['clear_dist']+2)
        margin = int(margin + (margin % 2))  # make it even
        self.im_size = setup_params['H']+margin*2
        self.crop_H = setup_params['H']
        self.crop_W = setup_params['W']
        self.fd_flag = setup_params['fd_flag']
        print(f'fd_flag: {self.fd_flag}')
        self.M = setup_params['M']  # magnification
        self.pixel_size_CCD = setup_params['pixel_size_CCD']
        self.device = setup_params['device']
        self.crop_r_start, self.crop_c_start = margin, margin
        self.setup_params = setup_params


    def forward(self, local_xyz, xy_center, Nphotons):
        """ generate one sub image each time around the xy_center
        model-free image generator
        :param local_xyz: ndarray, rank3, [1, ~, 3]
        :param xy_center: tuple, (x, y)
        :param Nphotons: ndarray, rank2, [1, ~]
        :return: one training image
        """
        # global xy, for interpolation
        xyz_global = local_xyz.copy()
        xyz_global[:, :, 0] = xyz_global[:, :, 0] + xy_center[0]
        xyz_global[:, :, 1] = xyz_global[:, :, 1] + xy_center[1]

        # form the image in a larger grid
        psfs = np.zeros((self.im_size, self.im_size))
        # psfs_ = np.zeros((self.im_size, self.im_size))
        # combine many PSFs into one image slightly bigger than the training image
        for ii in range(xyz_global.shape[1]):
            x_interp = xyz_global[0, ii, 0]  # um
            y_interp = xyz_global[0, ii, 1]  # um
            z_interp = xyz_global[0, ii, 2]  # um
            n_photon = Nphotons[0, ii]

            if self.fd_flag:
                psf = self.psf_module(x_interp, y_interp, z_interp)  # xy and rc, confusing!  add blur??
            else:
                psf = self.psf_module(0, 0, z_interp)  # only use the one xy position, here the center
            # show_imset((self.psf_module(-1, -1, -1), self.psf_module(-5, -1, 0), self.psf_module(5, 5, 0.5), self.psf_module(5, 0, 1)))

            # psf_ = np.pad(psf, int((self.im_size-self.interpolated_psf_width)/2))
            # lx_ccd_ = local_xyz[0, ii, 0] * self.M / self.pixel_size_CCD
            # ly_ccd_ = local_xyz[0, ii, 1] * self.M / self.pixel_size_CCD
            # psf_ = np.abs(scipy.ndimage.shift(psf_, (ly_ccd_, lx_ccd_)))
            # psf_ = psf_ / psf_.sum() * n_photon
            # psfs_ = psfs_+psf_

            # psf = psf / psf.sum() * n_photon  # photon normalization
            # lx_ccd = local_xyz[0, ii, 0] * self.M / self.pixel_size_CCD
            # ly_ccd = local_xyz[0, ii, 1] * self.M / self.pixel_size_CCD
            # r_start = int(np.round(ly_ccd + self.im_size / 2 - self.interpolated_psf_width / 2))
            # r_end = r_start + self.interpolated_psf_width
            # c_start = int(np.round(lx_ccd + self.im_size / 2 - self.interpolated_psf_width / 2))
            # c_end = c_start + self.interpolated_psf_width
            # if r_start < 0 or c_start < 0 or r_end > self.im_size or c_end > self.im_size:
            #     raise Exception(f'OUT OF RANGE!')
            # psfs[r_start: r_end, c_start: c_end] = psfs[r_start: r_end, c_start: c_end] + psf

            lx_ccd = local_xyz[0, ii, 0] * self.M / self.pixel_size_CCD
            ly_ccd = local_xyz[0, ii, 1] * self.M / self.pixel_size_CCD
            r_start_ = ly_ccd + self.im_size / 2 - self.interpolated_psf_width / 2
            r_start = int(np.floor(ly_ccd + self.im_size / 2 - self.interpolated_psf_width / 2))
            r_shift = r_start_ - r_start
            r_end = r_start + self.interpolated_psf_width
            c_start_ = lx_ccd + self.im_size / 2 - self.interpolated_psf_width / 2
            c_start = int(np.floor(lx_ccd + self.im_size / 2 - self.interpolated_psf_width / 2))
            c_shift = c_start_ - c_start
            c_end = c_start + self.interpolated_psf_width

            psf = np.abs(scipy.ndimage.shift(psf, (r_shift, c_shift)))

            psf = psf / psf.sum() * n_photon
            if r_start < 0 or c_start < 0 or r_end > self.im_size or c_end > self.im_size:
                raise Exception(f'OUT OF RANGE!')
            psfs[r_start: r_end, c_start: c_end] = psfs[r_start: r_end, c_start: c_end] + psf

            # blur can be added here
            pass

        # crop the image
        im = psfs[self.crop_r_start: self.crop_r_start+self.crop_H, self.crop_c_start: self.crop_c_start+self.crop_W]
        # show_imset(im)
        # add noise
        im_4D_tensor = torch.tensor(im, device=self.device).unsqueeze(0).unsqueeze(0)
        im_noisy = self.noise_layer(im_4D_tensor)  # background, poisson, read noise
        im_np = im_noisy[0, 0, :, :].cpu().numpy()

        # 01 projection
        if self.setup_params['project_01'] is False:
            im_np = (im_np - self.setup_params['global_factors'][0]) / self.setup_params['global_factors'][1]
        else:
            im_np = (im_np-im_np.min())/(im_np.max()-im_np.min())

        if (self.setup_params['nsig_unif'] is False) and (local_xyz.shape[1] > 1):  # remove emitters with fewer photons
            Nphotons = np.squeeze(Nphotons)
            local_xyz = local_xyz[:, Nphotons > self.setup_params['nsig_thresh'], :]

        return im_np, local_xyz


# generate training data (either images for localization cnn or locations and intensities for psf learning)
def gen_data(setup_params):

    # random seed for repeatability
    torch.manual_seed(999)
    np.random.seed(566)

    # calculate on GPU if available
    device = setup_params['device']
    torch.backends.cudnn.benchmark = True

    # calculate the effective sensor pixel size taking into account magnification and set the recovery pixel size to be
    # the same such that sampling of training positions is performed on this coarse grid
    setup_params['pixel_size_FOV'] = setup_params['pixel_size_CCD'] / setup_params['M']  # in [um]
    setup_params['pixel_size_rec'] = setup_params['pixel_size_FOV'] / 1  # in [um]

    # calculate the axial range and the axial pixel size depending on the volume discretization
    setup_params['axial_range'] = setup_params['zmax'] - setup_params['zmin']  # [um]
    setup_params['pixel_size_axial'] = setup_params['axial_range'] / setup_params['D']  # [um]

    # training data folder for saving
    path_train = setup_params['training_data_path']
    if not (os.path.isdir(path_train)):
        os.mkdir(path_train)  # creat the folder

    # print status
    print('=' * 50)
    print('Sampling examples for training')
    print('=' * 50)

    # batch size for generating training examples
    batch_size_gen = 1
    setup_params['batch_size_gen'] = batch_size_gen  # this should go before the training_image_layer
    ntrain_batches = int(setup_params['ntrain'] / batch_size_gen)
    setup_params['ntrain_batches'] = ntrain_batches

    ##### core of the project
    training_image_layer = TrainingImageLayer(setup_params)  # apply the PSF generator to get an image generator
    ####

    # generate examples for training
    labels_dict = {}
    for i in range(ntrain_batches):
        # sample a training example
        xyz, Nphotons = generate_batch(batch_size_gen, setup_params)  # for only one small sub-area, xyz is local, transform to global later
        if setup_params['fd_flag']:  # field-dependent mode
            # randomly choose one center
            center_idx = np.random.randint(0, setup_params['xy_centers_num'])
            xy_center = setup_params['xy_centers'][center_idx]  # to get global xyz from local xyz
        else:
            center_idx = 0  # doesn't matter
            xy_center = (0, 0)
        cropped_image, valid_local_xyz = training_image_layer(xyz, xy_center, Nphotons)  # get one training pair

        if setup_params['visualize']:
            # plot the image with xy and the simulated xy center on the top
            fig1 = plt.figure(1)
            imfig = plt.imshow(cropped_image, cmap='gray')
            if valid_local_xyz is not None:
                xyz2 = np.squeeze(valid_local_xyz, 0)
                show_xy = xyz2[:, 0:2]
                show_x = show_xy[:, 0] * setup_params['M'] / setup_params['pixel_size_CCD'] + setup_params['W']/2
                show_y = show_xy[:, 1] * setup_params['M'] / setup_params['pixel_size_CCD'] + setup_params['H']/2
                plt.plot(show_x, show_y, 'r+')
            plt.title(f'i={i}, xy_center: {xy_center}')
            plt.show()

            # fig1.colorbar(imfig)
            # plt.draw()
            # plt.pause(0.05)
            # plt.clf()

        # save training data
        im_name_tiff = path_train + 'im' + str(i).zfill(5) + '.tiff'
        io.imsave(im_name_tiff, cropped_image, check_contrast=False)

        labels_dict[str(i).zfill(5)] = (valid_local_xyz, center_idx)  # save the ground localization and crop center
        if i % np.round(ntrain_batches/90) == 0:
            print('Training Example [%d / %d]' % (i + 1, ntrain_batches))

    # calculate training set mean and standard deviation
    train_stats = CalcMeanStd_All(path_train, labels_dict)
    setup_params['train_stats'] = train_stats

    # print status
    print('=' * 50)
    print('Sampling examples for validation')
    print('=' * 50)

    # calculate the number of training batches to sample
    nvalid_batches = int(setup_params['nvalid'] // batch_size_gen)
    setup_params['nvalid_batches'] = nvalid_batches

    # set the number of particles to the middle of the range for validation
    num_particles_range = setup_params['num_particles_range']
    setup_params['num_particles_range'] = [num_particles_range[1]//2, num_particles_range[1]//2 + 1]

    # sample validation examples
    for i in range(nvalid_batches):
        # sample a training example
        xyz, Nphotons = generate_batch(batch_size_gen, setup_params)
        if setup_params['fd_flag']:
            center_idx = np.random.randint(0, setup_params['xy_centers_num'])
            xy_center = setup_params['xy_centers'][center_idx]
        else:
            center_idx = 0  # doesn't matter
            xy_center = (0, 0)
        cropped_image, valid_local_xyz = training_image_layer(xyz, xy_center, Nphotons)

        im_name_tiff = path_train + 'im' + str(i+ntrain_batches).zfill(5) + '.tiff'
        io.imsave(im_name_tiff, cropped_image, check_contrast=False)

        labels_dict[str(i + ntrain_batches).zfill(5)] = (valid_local_xyz, center_idx)  # save the ground localization and crop center
        if i % (np.round(nvalid_batches/10)) == 0:
            print('Validation Example [%d / %d]' % (i + 1, nvalid_batches))


    # save all xyz's dictionary as a pickle file
    path_labels = path_train + 'labels.pickle'
    with open(path_labels, 'wb') as handle:
        pickle.dump(labels_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # set the number of particles back to the specified range
    setup_params['num_particles_range'] = num_particles_range

    # partition built in simulation
    ind_all = np.arange(0, ntrain_batches + nvalid_batches, 1)
    list_all = ind_all.tolist()
    list_IDs = [str(i).zfill(5) for i in list_all]
    train_IDs = list_IDs[:ntrain_batches]
    valid_IDs = list_IDs[ntrain_batches:]
    partition = {'train': train_IDs, 'valid': valid_IDs}
    setup_params['partition'] = partition

    # update recovery pixel in xy to be x4 smaller if we are learning a localization net
    setup_params['pixel_size_rec'] = setup_params['pixel_size_FOV'] / 4  # in [um]

    # save setup parameters dictionary for training and testing
    path_setup_params = path_train + 'setup_params.pickle'
    with open(path_setup_params, 'wb') as handle:
        pickle.dump(setup_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print status
    print('Finished sampling examples!')
    # close figure if it was open for visualization
    if setup_params['visualize']:
        plt.close(fig1)


if __name__ == '__main__':

    # start a parser
    parser = argparse.ArgumentParser()

    # previously wrapped settings dictionary
    parser.add_argument('--setup_params', help='path to the parameters wrapped in the script parameter_setting', required=True)

    # parse the input arguments
    args = parser.parse_args()

    # run the data generation process
    gen_data(args.setup_params)
