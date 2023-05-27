
import numpy as np
from math import pi
import torch.nn as nn
from scipy import interpolate
import itertools
from sklearn.decomposition import PCA
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
from skimage import io
from skimage.morphology import erosion, dilation
import os
import glob
from helper_utils_ import show_imset

def polynomial_surface_fitting(data, order, xy):
    """
    Polynomial surface fitting
    :param data: ndarray, rank 2, [~,3], the first, second and third column corresponds to x, y and z respectively
    :param order: int, 1, 2, 3, or 4
    :param xy: ndarray, rank 2, [~, 2], the first, second column are x and y coordinates to predict
    :return: ndarray, rank1, [~,], ? predicted z at (x_col, y_col)
    """
    x, y = xy[:, 0], xy[:, 1]
    xdata, ydata, zdata = data[:, 0], data[:, 1], data[:, 2]
    constant = np.ones(data.shape[0])
    if order == 1:
        A = np.c_[constant, xdata, ydata]
        C, _, _, _ = scipy.linalg.lstsq(A, zdata)
        z = np.dot(np.c_[np.ones(x.shape[0]), x, y], C)
        return z
    elif order==2:
        A = np.c_[constant, xdata, ydata, xdata*ydata, xdata**2, ydata**2]
        C, _, _, _ = scipy.linalg.lstsq(A, zdata)
        z = np.dot(np.c_[np.ones(x.shape[0]), x, y, x*y, x**2, y**2], C)
        return z
    elif order==3:
        A = np.c_[constant, xdata, ydata, xdata * ydata, xdata ** 2, ydata ** 2, xdata*ydata**2, xdata**2*ydata,
                  xdata**3, ydata**3]
        C, _, _, _ = scipy.linalg.lstsq(A, zdata)
        z = np.dot(np.c_[np.ones(x.shape[0]), x, y, x * y, x ** 2, y ** 2, x*y**2, x**2*y, x**3, y**3], C)
        return z
    elif order==4:
        A = np.c_[constant, xdata, ydata, xdata * ydata, xdata ** 2, ydata ** 2, xdata * ydata ** 2, xdata ** 2 * ydata,
                  xdata ** 3, ydata ** 3, xdata*ydata**3, xdata**2*ydata**2, xdata**3*ydata, xdata**4, ydata**4]
        C, _, _, _ = scipy.linalg.lstsq(A, zdata)
        z = np.dot(np.c_[np.ones(x.shape[0]), x, y, x * y, x ** 2, y ** 2, x * y ** 2, x ** 2 * y, x ** 3, y ** 3,
                   x*y**3, x**2*y**2, x**3*y, x**4, y**4], C)
        return z
    elif order==5:
        A = np.c_[constant,
                  xdata, ydata,
                  xdata * ydata, xdata ** 2, ydata ** 2,
                  xdata * ydata ** 2, xdata ** 2 * ydata, xdata ** 3, ydata ** 3,
                  xdata * ydata ** 3, xdata ** 2 * ydata ** 2, xdata ** 3 * ydata, xdata ** 4, ydata ** 4,
                  ydata**5, xdata*ydata**4, xdata**2*ydata**3, xdata**3*ydata**2, xdata**4*ydata**1, xdata ** 5]
        C, _, _, _ = scipy.linalg.lstsq(A, zdata)
        z = np.dot(np.c_[np.ones(x.shape[0]), x, y, x * y, x ** 2, y ** 2, x * y ** 2, x ** 2 * y, x ** 3, y ** 3,
                         x * y ** 3, x ** 2 * y ** 2, x ** 3 * y, x ** 4, y ** 4,
                         y ** 5, x*y ** 4, x**2*y ** 3, x**3*y ** 2, x**4*y ** 1, x**5], C)

        return z
    else:
        print('Order should be smaller than 4')


def noise_processing(im, corner_size=10, thr=2.0):
    """
    remove part of noise in a measured image
    :param im: image
    :param corner_size: pixels at each corner as noise references
    :param thr: values below thr*standard deviation are set to 0
    :return: the processed image
    """
    tmp = np.concatenate(
        (np.concatenate((im[:corner_size, :corner_size], im[-corner_size:, :corner_size]), axis=0),
         np.concatenate((im[:corner_size, -corner_size:], im[-corner_size:, -corner_size:]), axis=0)),
        axis=1)
    mean_value = np.mean(tmp)
    std_value = np.std(tmp)
    im0 = im-mean_value
    mask = (im0 > thr*std_value)
    cross = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]])
    mask = erosion(mask, cross)
    mask = dilation(mask, cross)
    im1 = im0*mask
    im1 = im1.astype(tmp.dtype)
    return im1


def FDPSF_data_reader(data_folder):
    """
    Read field-dependent PSF data, have to include PSF stacks (.tif),
    xy_pos (.mat file, lateral positions of each PSF stack), pixel locations
    and nfp_pos (.mat file, z-scanning positions)

    :param data_folder: the folder the three types of files above
    :return: all PSFs (4D ndarray, num_xy*num_nfp*W*H), xy_pos, ndarray, rank 2, num_xy*2, nfp_pos (1, 29)
    """

    xy_pos = sio.loadmat(os.path.join(data_folder, 'xy_pos.mat'))['xy_pos'].astype(np.double)  # np, 2
    nfp_pos = sio.loadmat(os.path.join(data_folder, 'nfp_pos.mat'))['nfp_pos']  # 1, 29
    nfp_pos = nfp_pos[0, :]

    im_names = glob.glob(data_folder + '*.tif')
    if len(im_names) != xy_pos.shape[0]:
        raise Exception('Mismatch between PSF stacks and xy positions.')

    all_images = []
    for i, tif_file in enumerate(im_names):
        ims0 = io.imread(tif_file)  # uint16, 29x80x80
        ims1 = np.zeros(ims0.shape)
        for j in range(ims0.shape[0]):
            im0 = ims0[j, :, :]
            im1 = noise_processing(im0, thr=1.0)  # remove some noise
            minv, maxv = im1.min(), im1.max()
            im2 = (im1 - minv) / (maxv - minv)  # normalize to [0, 1]
            ims1[j, :, :] = im2
        all_images.append(ims1)
    all_images = np.array(all_images).astype(dtype=np.double)  # np, 29, 81, 81
    return all_images, xy_pos, nfp_pos


class FDPSFGenerator0(nn.Module):
    def __init__(self, all_psfs, xy_pos, z_pos):
        """
        PSF interpolation/fitting class
        :param all_psfs: ndarray, rank 4, [xy_num, z_num, psf_size, psf_size]
        :param xy_pos: ndarray, rank 2, [xy_num, 2], phisical coordinates at the object plane
        :param z_pos: ndarray, rank 1, [z_num, ], z means nfp or z of emitters
        """
        super().__init__()
        self.all_psfs = all_psfs  #
        self.xy_num, self.z_num, self.psf_size = all_psfs.shape[0], all_psfs.shape[1], all_psfs.shape[2:]
        self.pca_variance = 0.95  # the amount of explained variance in PCA
        self.pca_list = []  # PCA object at all axial planes
        self.pca_X_new_list = []  # new data after dimensionality reduction in another space
        for i in range(all_psfs.shape[1]):
            X = all_psfs[:, i, :, :].reshape(self.xy_num, -1)
            pca = PCA(n_components=self.pca_variance)
            X_new = pca.fit_transform(X)
            self.pca_list.append(pca)
            self.pca_X_new_list.append(X_new)
        self.xy_pos = xy_pos
        if xy_pos.shape[0] != self.xy_num:
            raise Exception("Sorry, xy number mismatch in psf data and position data!")
        self.x_range = [np.min(xy_pos[:, 0]), np.max(xy_pos[:, 0])]  # pixel coordinates at the image plane
        self.y_range = [np.min(xy_pos[:, 1]), np.max(xy_pos[:, 1])]  # pixel coordinates at the image plane
        self.z_pos = z_pos
        self.z_range = [np.min(z_pos), np.max(z_pos)]
        self.num_pick1 = 7  # first choosing range in lateral point selection
        self.num_pick2 = 13  # second choosing range in lateral point selection
        self.fitting_num_xy = 3  # point number in fitting in a lateral plane, 2D
        self.fitting_num_z = 2  # point number in fitting along z, 1D
        self.fitting_order = 1
        self.nn_flag = True  # PSF's non-negative flag


        self.max_r = np.max(np.sqrt(np.sum(xy_pos**2, axis=1)))

        print(f'x range in calibration: {self.x_range} [um]')
        print(f'y range in calibration: {self.y_range} [um]')
        print(f'nfp range in calibration: {self.z_range} [um]')
        print(f'maximum radius: {self.max_r} [um]')

    def fitting_xy(self, c, xys, xy, order):  # 2D fitting
        """
        2D fitting regarding c (coefficients after PCA, data X in new space)
        :param c: ndarray, rank 2, [n_samples, n_features], coefficients
        :param xys: ndarray, rank 2, [n_samples, 2], known/selected xy coordinates
        :param xy: ndarray, rank 2, [1, 2], prediction position, one at each time
        :param order: int, fitting order, from 1 to 4
        :return: c_fitted, ndarray, rank2, [1, n_features]
        """

        c_fitted = np.zeros((1, c.shape[1]))
        for col_idx in range(c.shape[1]):
            data = np.c_[xys, c[:, col_idx]]
            z = polynomial_surface_fitting(data, order, xy)
            c_fitted[0, col_idx] = z

            # # show interpolation details
            # xys_ = np.concatenate((xys, xy), axis=0)
            # xnew = np.linspace(np.min(xys_[:, 0]), np.max(xys_[:, 0]), 10)
            # ynew = np.linspace(np.min(xys_[:, 1]), np.max(xys_[:, 1]), 10)
            # X, Y = np.meshgrid(xnew, ynew)
            # XX, YY = X.flatten(), Y.flatten()
            # znew = polynomial_surface_fitting(data, order, np.c_[XX, YY])
            # Z = znew.reshape(X.shape)
            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')
            # ax.plot_wireframe(X, Y, Z)
            # ax.scatter(data[:, 0], data[:, 1], data[:, 2], 'o', color='k', s=48)
            # ax.scatter(xy[0, 0], xy[0, 1], z, marker='*', color='green', s=58)
            # plt.show()

        return c_fitted

    def interpolation_axial(self, c, zs, z):
        """
        1D interpolation along axis
        :param c: ndarray, rank 2, [n_samples, n_features], coefficients
        :param zs: ndarray, rank1, [num_z, ], known z positions
        :param z: scalar, z for prediction,
        :return: interpolated c, ndarray, rank 2, [1, n_features], one at each time
        """
        c_interp = np.zeros((1, c.shape[1]))
        for col_idx in range(c.shape[1]):
            c_col = c[:, col_idx]
            f = interpolate.interp1d(zs, c_col)
            c_ = f(z)
            c_interp[0, col_idx] = c_

            # # show interpolation details
            # xnew = np.linspace(zs.min(), zs.max(), 20)
            # ynew = f(xnew)
            # plt.figure()
            # plt.plot(xnew, ynew)
            # plt.scatter(zs, c_col, marker='v')
            # plt.scatter(z, c_, marker='*')
            # # plt.axis('equal')
            # plt.show()

        return c_interp

    def select_points(self, xy):
        """
        find calibration points around target position xy
        :param xy: ndarray, rank 2, [1,2], target xy position
        :return: tuple, [num_points, ], indices of chosen points
        """
        distances_xy = np.sum((self.xy_pos - xy) ** 2, axis=1)  # distances from target xy to all calibration points
        self.min_idx = np.argmin(distances_xy)  # find the index of the closest point, for possible negative cases later
        indices_xy = np.argpartition(distances_xy,
                                     self.num_pick1)[:self.num_pick1]  # first selection regarding distance

        # # show the first choice
        # xys1 = self.xy_pos[indices_xy, :]
        # plt.figure()
        # plt.scatter(self.xy_pos[:, 0], self.xy_pos[:, 1], marker='*')  # all the points
        # plt.scatter(xy[0, 0], xy[0, 1], marker='o')  # target position
        # plt.scatter(xys1[:, 0], xys1[:, 1], marker='v')  # firstly chosen points
        # plt.axis('equal')
        # plt.show()

        point_num = self.fitting_num_xy
        all_combinations = list(itertools.combinations(indices_xy, point_num))  # choose
        angle_variance_list = []
        for combination in all_combinations:
            vectors = self.xy_pos[combination, :] - xy
            angles = np.sort(np.angle(vectors[:, 0] + 1j*vectors[:, 1]))  # [-pi, pi]
            target_mean_angle = 2*pi/point_num
            angle_variance = np.sum(((angles[1:]-angles[:-1])-target_mean_angle)**2)/(point_num-1)
            angle_variance_list.append(angle_variance)
        final_idx = all_combinations[np.argmin(angle_variance_list)]  # choose regarding angle variance/variation

        xys1 = self.xy_pos[final_idx, :]
        xmin, xmax = xys1[:, 0].min(), xys1[:, 0].max()
        ymin, ymax = xys1[:, 1].min(), xys1[:, 1].max()
        if xy[0, 0] > xmin and xy[0, 0] < xmax and xy[0, 1] > ymin and xy[0, 1] < ymax:  # is it a good selection?
            pass  # yes
        else:  # no, increase searching range
            indices_xy = np.argpartition(distances_xy, self.num_pick2)[:self.num_pick2]
            all_combinations = list(itertools.combinations(indices_xy, point_num))  # choose
            angle_variance_list = []
            for combination in all_combinations:
                vectors = self.xy_pos[combination, :] - xy
                angles = np.sort(np.angle(vectors[:, 0] + 1j * vectors[:, 1]))  # [-pi, pi]
                target_mean_angle = 2 * pi / point_num
                angle_variance = np.sum(((angles[1:] - angles[:-1]) - target_mean_angle) ** 2) / (point_num - 1)
                angle_variance_list.append(angle_variance)
            final_idx = all_combinations[np.argmin(angle_variance_list)]

        # # show the second choice
        # xys1 = self.xy_pos[final_idx, :]
        # plt.figure()
        # plt.scatter(self.xy_pos[:, 0], self.xy_pos[:, 1], marker='*')  # all the points
        # plt.scatter(xy[0, 0], xy[0, 1], marker='o')  # target position
        # plt.scatter(xys1[:, 0], xys1[:, 1], marker='v')  # firstly chosen points
        # plt.axis('equal')
        # plt.show()

        return final_idx

    def forward(self, x, y, z):
        """
        PSF interpolation
        :param x: scalar, pixel coordinate at image plane
        :param y: scalar, pixel coordinate at image plane
        :param z: scalar, axial coordinate for object, z of an emitter or nfp, unit: um
        :return: ndarray, rank2, [psf_size, psf_size], interpolated PSF
        """
        # find xy positions used for fitting, how many laterally and axially
        target_xy = np.array([[x, y]])
        indices_xy = self.select_points(target_xy)  # It matters a lot
        xys = self.xy_pos[indices_xy, :]

        distances_z = (self.z_pos-z)**2
        indices_z = np.argpartition(distances_z, self.fitting_num_z)[:self.fitting_num_z]
        zs = self.z_pos[indices_z]

        # # show final point selection
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # xy_ = np.repeat(xys, zs.shape[0], axis=0)
        # nn = zs[:, np.newaxis]
        # nfps_ = np.repeat(nn, xys.shape[0], axis=1).flatten('F')
        # ax.scatter(xy_[:, 0], xy_[:, 1], nfps_, marker='v', s=20, label='Selected calibration points')
        # ax.scatter(x, y, z, marker='o', s=20, label='Target point')
        # ax.legend()
        # plt.show()

        psfs_axial = np.zeros((self.fitting_num_z, self.psf_size[0]**2))
        show1 = []
        for idx, idx_z in enumerate(indices_z):
            # psfs = self.all_psfs[indices_xy, idx_z, :, :].reshape(len(indices_xy), -1)
            c = self.pca_X_new_list[idx_z][indices_xy, :]

            # psfs = self.all_psfs[indices_xy, idx, :, :]
            # show1.append(psfs)
            # title = str(indices_xy[0]) +'-'+ str(indices_xy[1]) +'-'+ str(indices_xy[2])
            # print(title)
            # show_imset((psfs[0,:,:], psfs[1,:,:], psfs[2,:,:]))

            # psfs = psfs.reshape(self.fitting_num_xy, -1).transpose()
            # u, c = self.svd(psfs)

            # u_show = u.reshape(self.psf_size[0], self.psf_size[1], 3)
            # show_imset((u_show[:,:,0], u_show[:,:,1], u_show[:,:,2]))


            c_fitted = self.fitting_xy(c, xys, target_xy, self.fitting_order)
            # c_fitted = self.pca_X_new_list[idx_z][121, :][np.newaxis, :]


            fitted_psf = self.pca_list[idx_z].inverse_transform(c_fitted)


            # c_fitted = self.fitting_xy(c[0:1, :], xys, np.array([x]), np.array([y]), 2)
            # fitted_psf = (u[:, 0:c_fitted.shape[0]] @ c_fitted)

            psfs_axial[idx, :] = fitted_psf[0, :]

        ## show interpolated psfs at each chosen nfp
        # imss = psfs_axial.reshape(self.fitting_num_z, self.psf_size[0], self.psf_size[1])
        # show2 = imss

        # non-negativity test
        if np.abs(psfs_axial.min()) > 0.1 * psfs_axial.max():  # fail
            self.nn_flag = False
            psfs_axial = self.all_psfs[self.min_idx, indices_z, :, :].reshape(len(indices_z), -1)

        else:
            self.nn_flag = True

        pca_z = PCA()
        c = pca_z.fit_transform(psfs_axial)
        c_interpolated = self.interpolation_axial(c, zs, z)  # axial interpolation
        psf = pca_z.inverse_transform(c_interpolated).reshape(self.psf_size)
        psf = np.abs(psf)  # make it non-negative

        # chosen_psfs1 = self.all_psfs[indices_xy, indices_z[0], :, :]
        # chosen_psfs2 = self.all_psfs[indices_xy, indices_z[1], :, :]
        # show_imset((chosen_psfs1[0,:,:], chosen_psfs1[1,:,:], chosen_psfs1[2,:,:],
        #             chosen_psfs2[0, :,:], chosen_psfs2[1, :,:], chosen_psfs2[2,:,:],
        #             show2[0,:,:], show2[1,:,:], psf), title=str(indices_xy[0]+1) +'-'+ str(indices_xy[1]+1) +'-'+ str(indices_xy[2]+1))

        return psf



def image_generation(xyzps, image_size, setup_params):
    """
    generate one image according to a given xyz-photon list
    :param xyzps: xyz-photon list, ndarray, rank 2, ~*4
    :param image_size: size in pixel, int, larger than the given xy boundary, odd
    :param setup_params: all kinds of parameters
    :return: an image
    """
    psf_generator = setup_params['psf_generator']

    xy_range = np.amax(np.abs(xyzps[:, 0:2])) * 2
    pixel_size_FOV = setup_params['pixel_size_CCD']/setup_params['M']
    imsize_before_cropping = np.ceil(xy_range/pixel_size_FOV + setup_params['interpolated_psf_width'] + 1)
    imsize_before_cropping = int(imsize_before_cropping + (imsize_before_cropping % 2 + 1))  # make it odd

    interpolated_psf_width = setup_params['interpolated_psf_width']

    if image_size > imsize_before_cropping:  #imsize_before_cropping is the 'critical' size to cover all the xy positions
        imsize_before_cropping = image_size

    im = np.zeros((imsize_before_cropping, imsize_before_cropping))
    for i in range(xyzps.shape[0]):
        x, y, z, nphotons = xyzps[i, :]
        psf = psf_generator(x, y, z)


        # psf = np.pad(psf, int((image_size - interpolated_psf_width) / 2))
        # psf = psf / psf.sum() * nphotons
        # lx_ccd = x / pixel_size_FOV
        # ly_ccd = y / pixel_size_FOV
        # psf = np.abs(scipy.ndimage.shift(psf, (ly_ccd, lx_ccd)))
        # im = im + psf



        lx_ccd = x / pixel_size_FOV
        ly_ccd = y / pixel_size_FOV
        r_start_ = ly_ccd + imsize_before_cropping / 2 - setup_params['interpolated_psf_width'] / 2
        r_start = int(np.floor(ly_ccd + imsize_before_cropping / 2 - setup_params['interpolated_psf_width'] / 2))
        r_shift = r_start_-r_start
        r_end = r_start + setup_params['interpolated_psf_width']
        c_start_ = lx_ccd + imsize_before_cropping / 2 - setup_params['interpolated_psf_width'] / 2
        c_start = int(np.floor(lx_ccd + imsize_before_cropping / 2 - setup_params['interpolated_psf_width'] / 2))
        c_shift = c_start_-c_start
        c_end = c_start + setup_params['interpolated_psf_width']

        psf = np.abs(scipy.ndimage.shift(psf, (r_shift, c_shift)))
        psf = psf / psf.sum() * nphotons

        if r_start < 0 or c_start < 0 or r_end > imsize_before_cropping or c_end > imsize_before_cropping:
            raise Exception(f'OUT OF RANGE!')
        im[r_start: r_end, c_start: c_end] = im[r_start: r_end, c_start: c_end] + psf




    # crop according to image_size if image_size is smaller than imsize_before_cropping
    rc_start = int((imsize_before_cropping-image_size)/2)  # both are odd, so the result is an integer
    im_final = im[rc_start: rc_start+image_size, rc_start: rc_start+image_size]

    return im_final


def interpolation_axial(c, zs, z):
    """
    1D interpolation along z axis
    :param c: ndarray, rank 2, [n_samples/num_z, n_features], coefficients
    :param zs: ndarray, rank1, [num_z, ], known z positions
    :param z: scalar, z for prediction,
    :return: interpolated c, ndarray, rank 2, [1, n_features], one at each time
    """
    c_interp = np.zeros((1, c.shape[1]))
    for col_idx in range(c.shape[1]):
        c_col = c[:, col_idx]
        f = interpolate.interp1d(zs, c_col, kind='cubic')
        y = f(z)
        c_interp[0, col_idx] = y

        # show interpolation details
        # xnew = np.linspace(zs.min(), zs.max(), 200)
        # ynew = f(xnew)
        # plt.figure()
        # plt.plot(xnew, ynew)
        # plt.scatter(zs, c_col, marker='v')
        # plt.scatter(z, y, marker='*')
        # plt.show()

    return c_interp


def corr2(A, B):
    """
    2D correlation coefficient, from corr2 of Matlab
    :param A: ndarray, rank2
    :param B: ndarray, rank2
    :return: correlation coefficient
    """
    A_mean = np.mean(A)
    B_mean = np.mean(B)
    r = np.sum((A-A_mean)*(B-B_mean))/np.sqrt(np.sum((A-A_mean)**2)*np.sum((B-B_mean)**2))
    return r


def rmse(A, B):
    return np.sqrt(np.sum((A-B)**2))




