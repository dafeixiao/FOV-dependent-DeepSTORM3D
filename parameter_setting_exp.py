
# for field-dependent DeepSTORM3D
from math import pi
import os
import numpy as np
from PSFI3D.PSF_interpolation import FDPSFGenerator0
import torch
import scipy.io as sio

def parameter_setting_exp():
    # current directory
    path_curr_dir = os.getcwd()
    path_dict = {'path_curr_dir': path_curr_dir}

    # ======================================================================================
    # optics settings: objective, light, sensor properties
    # ======================================================================================
    # pixel_size_CCD = 11  # sensor pixel size in [um] (including binning)
    pixel_size_CCD = 13  # sensor pixel size in [um] (including binning)
    M = 100  # optical magnification
    # optical settings dictionary
    optics_dict = {'pixel_size_CCD': pixel_size_CCD, 'M': M}

    # single training image dimensions
    H, W = 161, 161  # in sensor [pixels]  # the size of sub-images, odd
    # safety margin from the boundary to prevent PSF truncation
    clear_dist = 20  # in sensor [pixels]
    # training z-range anf focus
    # zmin = -2.0  # minimal z in [um] (including the axial shift)
    # zmax = 1.0  # maximal z in [um] (including the axial shift)
    zmin = 0.0  # minimal z in [um] (including the axial shift)
    zmax = 4.0  # maximal z in [um] (including the axial shift)
    # discretization in z
    D = 81
    # data dimensions dictionary
    data_dims_dict = {'H': H, 'W': W, 'clear_dist': clear_dist, 'zmin': zmin, 'zmax': zmax, 'D': D}

    # ======================================================================================
    # number of emitters in each FOV
    # ======================================================================================
    # upper and lower limits for the number fo emitters
    num_particles_range = [1, 42]
    # number of particles dictionary
    num_particles_dict = {'num_particles_range': num_particles_range}

    # ======================================================================================
    # signal counts distribution and settings
    # ======================================================================================
    # boolean that specifies whether the signal counts are uniformly distributed
    nsig_unif = False
    # range of signal counts assuming a uniform distribution
    nsig_unif_range = [6000, 9000]  # in [counts]
    # parameters for sampling signal counts assuming a gamma distribution
    nsig_gamma_params = [3, 3000]  # in [counts]
    # threshold on signal counts to discard positions from the training labels
    nsig_thresh = 8000  # in [counts]
    # signal counts dictionary
    nsig_dict = {'nsig_unif': nsig_unif, 'nsig_unif_range': nsig_unif_range, 'nsig_gamma_params': nsig_gamma_params,
                 'nsig_thresh': nsig_thresh}

    # ======================================================================================
    # blur standard deviation for smoothing PSFs to match experimental conditions
    # ======================================================================================
    # upper and lower blur standard deviation for each emitter to account for finite size
    blur_std_range = [0.75, 1.25]  # in sensor [pixels], doesn't matter
    # blur dictionary
    blur_dict = {'blur_std_range': blur_std_range}

    # ======================================================================================
    # uniform/non-uniform background settings
    # ======================================================================================
    # uniform background value per pixel
    unif_bg = 30  # in [counts]
    # boolean flag whether or not to include a non-uniform background
    nonunif_bg_flag = False
    # maximal offset for the center of the non-uniform background in pixels
    nonunif_bg_offset = [10, 10]  # in sensor [pixels]
    # peak and valley minimal values for the super-gaussian; randomized with addition of up to 50%
    nonunif_bg_minvals = [20.0, 100.0]  # in [counts]
    # minimal and maximal angle of the super-gaussian for augmentation
    nonunif_bg_theta_range = [-pi/4, pi/4]  # in [radians]
    # nonuniform background dictionary
    nonunif_bg_dict = {'nonunif_bg_flag': nonunif_bg_flag, 'unif_bg': unif_bg, 'nonunif_bg_offset': nonunif_bg_offset,
                       'nonunif_bg_minvals': nonunif_bg_minvals, 'nonunif_bg_theta_range': nonunif_bg_theta_range}

    # ======================================================================================
    # read noise settings
    # ======================================================================================

    # boolean flag whether or not to include read noise
    read_noise_flag = True
    # flag whether of not the read noise standard deviation is not uniform across the FOV
    read_noise_nonuinf = False
    # range of baseline of the min-subtracted data in STORM
    read_noise_baseline_range = [36, 70]  # in [counts]
    # read_noise_baseline_range = [40, 41]
    # read noise standard deviation upper and lower range
    read_noise_std_range = [26, 40]  # in [counts]
    # read_noise_std_range = [40, 41]
    # read noise dictionary
    read_noise_dict = {'read_noise_flag': read_noise_flag, 'read_noise_nonuinf': read_noise_nonuinf,
                       'read_noise_baseline_range': read_noise_baseline_range,
                       'read_noise_std_range': read_noise_std_range}

    # ======================================================================================
    # image normalization settings
    # ======================================================================================
    # boolean flag whether or not to project the images to the range [0, 1]
    project_01 = True
    # global normalization factors for STORM (subtract the first and divide by the second)
    global_factors = [0.0, 250.0]  # in [counts]
    # image normalization dictionary
    norm_dict = {'project_01': project_01, 'global_factors': global_factors}

    # ======================================================================================
    # training data settings
    # ======================================================================================
    # number of training and validation examples
    ntrain = 9000*10
    nvalid = 1000*10
    # path for saving training examples: images + locations for localization net or locations + photons for PSF learning
    training_data_path = path_curr_dir + "/training_data/data_exp_fd/"
    # boolean flag whether to visualize examples while created
    visualize = False
    # training data dictionary
    training_dict = {'ntrain': ntrain, 'nvalid': nvalid, 'training_data_path': training_data_path, 'visualize': visualize}

    # ======================================================================================
    # learning settings
    # ======================================================================================
    # results folder to save the trained model
    results_path = path_curr_dir + "/training_results/results_exp_fd/"

    # maximal dilation flag when learning a localization CNN (set to None if learn_mask=True as we use a different CNN)
    dilation_flag = True  # if set to 1 then dmax=16 otherwise dmax=4
    # batch size for training a localization model (set to 1 for mask learning as examples are generated 16 at a time)
    batch_size = 8
    # maximal number of epochs
    max_epochs = 25
    # initial learning rate for adam
    initial_learning_rate = 0.0005
    # scaling factor for the loss function
    scaling_factor = 800.0
    # learning dictionary
    learning_dict = {'results_path': results_path, 'dilation_flag': dilation_flag, 'batch_size': batch_size,
                     'max_epochs': max_epochs, 'initial_learning_rate': initial_learning_rate,
                     'scaling_factor': scaling_factor}

    # ======================================================================================
    # resuming from checkpoint settings
    # ======================================================================================
    # boolean flag whether to resume training from checkpoint
    resume_training = False
    # number of epochs to resume training
    num_epochs_resume = None
    # saved checkpoint to resume from
    checkpoint_path = None
    # checkpoint dictionary
    checkpoint_dict = {'resume_training': resume_training, 'num_epochs_resume': num_epochs_resume,
                       'checkpoint_path': checkpoint_path}

    # ======================================================================================
    # final resulting dictionary including all parameters
    # ======================================================================================
    settings = {**path_dict, **num_particles_dict, **nsig_dict, **blur_dict, **nonunif_bg_dict, **read_noise_dict,
                **norm_dict, **optics_dict, **data_dims_dict, **training_dict, **learning_dict, **checkpoint_dict}


    # other required parameters

    psfs_1 = sio.loadmat('./PSF_Database/psf_stacks_1.mat')['psf_stacks_1']
    psfs_1 = psfs_1.astype(dtype=np.double)
    psfs_2 = sio.loadmat('./PSF_Database/psf_stacks_2.mat')['psf_stacks_2']
    psfs_2 = psfs_2.astype(dtype=np.double)
    psfs_3 = sio.loadmat('./PSF_Database/psf_stacks_3.mat')['psf_stacks_3']
    psfs_3 = psfs_3.astype(dtype=np.double)
    psfs_4 = sio.loadmat('./PSF_Database/psf_stacks_4.mat')['psf_stacks_4']
    psfs_4 = psfs_4.astype(dtype=np.double)
    all_images = np.concatenate((psfs_1, psfs_2, psfs_3, psfs_4), axis=1)


    for i in range(all_images.shape[0]):
        for j in range(all_images.shape[1]):
            im = all_images[i, j, :, :]
            minv, maxv = im.min(), im.max()
            im2 = (im - minv) / (maxv - minv)  # normalize to [0, 1]
            all_images[i, j, :, :] = im2
    all_images = all_images.astype(dtype=np.double)

    xy_pos = sio.loadmat('./PSF_Database' + '/xy_pos.mat')['xy_pos'].astype(np.double)
    z_pos = np.arange(0, 28) * 0.15
    pixel_size_FOV = pixel_size_CCD / M  # pixel size at the object plane
    xy_pos = xy_pos*pixel_size_FOV  # from pixel unit to physical unit
    psf_generator = FDPSFGenerator0(all_images, xy_pos, z_pos)  # get the PSF generator, important!
    settings['psf_generator'] = psf_generator
    print(f'get PSF generator!')
    settings['interpolated_psf_width'] = all_images.shape[-1]  # size of the cropped PSFs, pixel unit

    # 9 centers at the large FOV. you can also design other centers.
    num_W = 11  # the large field of view is divided into num_W-by-num_W sub areas, odd
    total_W = W + (num_W-1)*(W-2*clear_dist)  # the total FOV after stitching all the sub areas
    print(f'The final stitched image is a square with a width of {total_W} pixels.')

    xy_centers = []  # centers of each sub-area, in um
    xy_maps = []  # xy maps of each sub-area
    crop_rcs = []  # crop starts when transforming a big FOV into a series of sub-areas
    center_interval = W-2*clear_dist  # valid width of each sub area, in pixel unit
    coordx, coordy = np.meshgrid(np.linspace(-(W - 1) / 2, (W - 1) / 2, W) * pixel_size_FOV,
                                 np.linspace(-(H - 1) / 2, (H - 1) / 2, H) * pixel_size_FOV,
                                 indexing='xy')  # a set of basi xy map

    for i in range(num_W):
        for j in range(num_W):
            x = (j - np.floor(num_W/2)) * center_interval * pixel_size_FOV  # physical unit
            y = (i - np.floor(num_W/2)) * center_interval * pixel_size_FOV
            xy_centers.append((x, y))
            xmap, ymap = coordx+x, coordy+y
            xy_map = np.concatenate((xmap[np.newaxis, :, :], ymap[np.newaxis, :, :]), axis=0)
            xy_maps.append(xy_map)
            crop_rcs.append((i * center_interval, j * center_interval))  # cropping starts of each sub area

    settings['xy_centers'] = xy_centers
    settings['xy_centers_num'] = len(xy_centers)
    settings['xy_maps'] = xy_maps
    settings['center_interval'] = center_interval
    settings['total_W'] = total_W
    settings['crop_rcs'] = crop_rcs

    settings['fd_flag'] = True  # fd consideration flag
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    settings['device'] = device

    return settings

if __name__ == '__main__':
    parameters = parameter_setting_exp()

