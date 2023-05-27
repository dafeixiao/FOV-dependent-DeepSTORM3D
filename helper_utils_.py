
import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos, sin
import math
import PIL
from skimage.morphology import erosion, dilation
from mpl_toolkits import mplot3d
import scipy

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

def reshape_mask(mask, target_aperture):
    """
    resize phase mask according to a given aperture
    :param mask: centered phase matrix, ndarray, rank 2
    :param target_aperture: target aperture, 0-1 ndarray, rank 2
    :return: resized phase matrix, has the same aperture size as the target_aperture
    """
    Rc, Cc = target_aperture.shape
    RN, CN = mask.shape
    mask_mr = mask[int(np.ceil((RN-1)/2)), :]  # middle row
    mask_mc = mask[:, int(np.ceil((CN-1)/2))]  # middle column
    aper_mr = target_aperture[int(np.ceil((Rc - 1) / 2)), :]
    aper_mc = target_aperture[:, int(np.ceil((Cc - 1) / 2))]
    # for mask
    x1 = next((i for i, x in enumerate(mask_mr) if x), None)  # index of the first nonzero value
    x2 = next((i for i, x in enumerate(np.flipud(mask_mr)) if x), None)  # index of the last nonzero value, flipud!
    x2 = RN - x2 + 1
    y1 = next((i for i, x in enumerate(mask_mc) if x), None)
    y2 = next((i for i, x in enumerate(np.flipud(mask_mc)) if x), None)
    y2 = RN - y2 + 1
    # for aperture
    x1_C = next((i for i, x in enumerate(aper_mr) if x), None)
    x2_C = next((i for i, x in enumerate(np.flipud(aper_mr)) if x), None)
    x2_C = Rc - x2_C + 1
    y1_C = next((i for i, x in enumerate(aper_mc) if x), None)
    y2_C = next((i for i, x in enumerate(np.flipud(aper_mc)) if x), None)
    y2_C = Cc - y2_C + 1

    W = np.floor((x2_C - x1_C) / (x2 - x1) * RN)  # target width
    H = np.floor((y2_C - y1_C) / (y2 - y1) * CN)  # target height
    im = PIL.Image.fromarray(mask)
    mask_res = np.array(im.resize([int(H), int(W)], PIL.Image.BICUBIC))  # resized mask

    # pad or crop to get same shape as aperture
    R, C = mask_res.shape
    if R < Rc or C < Cc:  # padding
        mask = np.zeros((Rc, Cc))
        mask[int(np.round(Rc/2-R/2)):int(np.round(Rc/2-R/2))+R,
        int(np.round((Cc/2-C/2))):int(np.round((Cc/2-C/2)))+C] = mask_res
    elif R > Rc or C > Cc:
        mask = mask_res[int(np.round(R/2-Rc/2)):int(np.round(R/2-Rc/2))+Rc,
               int(np.round(C/2-Cc/2)):int(np.round(C/2-Cc/2))+Cc]

    return np.array(mask)


def show_imset(imset, title=None, rc=None):
    """
    show all the matrices (more than 1) in the imset
    :param imset: a tuple with images(ndarray or tensor)
    :param title: title of the imset
    :param rc: tuple to set row and column number
    :return: show them
    """
    fig = plt.figure(1)
    if type(imset) is not tuple:
        plt.imshow(imset)
    else:
        im_num = len(imset)
        if rc is None:
            r = int(np.floor(np.sqrt(im_num)))
            c = int(np.ceil(im_num/r))
        else:
            r, c = rc[0], rc[1]
        if title is not None:
            st = fig.suptitle(title, fontsize="x-large")
        for i in range(im_num):
            plt.subplot(r, c, i + 1)
            # plt.imshow(imset[i], cmap='gray')
            plt.imshow(imset[i])
            plt.axis('off')
            # plt.colorbar(shrink=0.5)
    plt.savefig('for_fig1.png', dpi=600, bbox_inches='tight')
    plt.show()


def circular_aper(N, d_ratio):
    """
    creat a circular aperture/window
    :param N: size(how many pixels) of the aperture
    :param d_ratio: diameter ratio, the maximum is 1
    :return: aperture, ndarray
    """
    xs = np.linspace(-1, 1, N)
    x, y = np.meshgrid(xs, xs, indexing='xy')
    r = np.sqrt(x**2 + y**2)
    aperture = r < (d_ratio)
    aperture = aperture.astype('float64')
    aperture[r == d_ratio] = 0.5
    return aperture


def square_window(N, d_ratio):
    """
    square window/aperture
    :param N: size of the square matrix
    :param d_ratio: ratio of diameter/width to N, the maximum is 1
    :return: window matrix W, ndarray
    """
    xs = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    X, Y = np.abs(X), np.abs(Y)
    W1 = (X < d_ratio).astype('float64')
    W1[X == d_ratio/2] = 0.5
    W2 = (Y < d_ratio).astype('float64')
    W2[Y == d_ratio / 2] = 0.5
    return W1*W2


def square_window2(N, rx, ry, d_ratio):
    """
    square window/aperture at (rx, ry). （0, 0） is the center.
    :param N: size of the square matrix
    :param rx and ry: relative coordinates[-1, 1]
    :param d_ratio: ratio of diameter/width to N
    :return: window matrix W, ndarray
    """
    xs = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    X = X - rx
    Y = Y - ry
    X, Y = np.abs(X), np.abs(Y)
    W1 = (X < d_ratio).astype('float64')
    W2 = (Y < d_ratio).astype('float64')

    return W1*W2


def phase2voltage(phase, mapping_curve):
    """
    transform phase mask to voltage mask, numpy
    :param phase: to be transformed, [-pi, pi]
    :param mapping_curve: calibration curve of a certain wavelength, rank 1
    :return: voltage mask [0, 255]
    """
    phase = phase + pi  # to [0, 2pi]
    r, c = phase.shape[0], phase.shape[1]
    voltage = np.zeros((r, c))
    for i in range(r):
        for j in range(c):
            pv = phase[i, j]
            ind = np.argmin(np.abs(mapping_curve-pv))
            voltage[i, j] = ind
    voltage = voltage.astype('uint8')
    return voltage


def vortex_phase(M, N, pn):
    """
    vortex phase generation
    :param M: row #
    :param N: column #
    :param pn: how many periods
    :return: vortex phase, nd array
    """
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, M)
    X, Y = np.meshgrid(x, y, indexing='xy')
    phase = np.angle(X + 1j*Y)
    phase = np.mod(phase, 2*pi/pn) * pn
    return phase


def Zernike_basis_numpy(aper_size, N, M=5):
    """
    aper_size: pixel number of the aperture
    N: size of the output, larger than the aper_size
    M: maximum order
    return: N N number (-1, 1), count
    order: zero, piston, tip, tilt, defocus, astigatism, astigmatism, coma, coma, trefoil, trefoil,
    spherical,....
    """

    x_lin = np.linspace(-1, 1, aper_size)  # create grid
    [X, Y] = np.meshgrid(x_lin, x_lin)

    r = np.sqrt(X ** 2 + Y ** 2)  # polar coordinates
    phi = np.arctan2(Y, X)

    mask_circ = (X ** 2 + Y ** 2) < 1
    count = 1
    D = np.zeros((N, N, M ** 2))
    for n in np.arange(0, M + 1):
        for m in np.arange(0, n + 1):
            Rmn = 0
            if (np.mod(n - m, 2) == 0):
                for k in np.arange(0, int((n - m) / 2) + 1):
                    Rmn = Rmn + ((-1) ** k) * (math.factorial(n - k)) * (r ** (n - 2 * k)) / \
                          ((math.factorial(k)) * (math.factorial(int((n + m) / 2) - k)) * (math.factorial(int((n - m)/2) - k)))
                if m == 0:
                    tmp = Rmn * mask_circ
                    pad_val = int(np.round((N - aper_size) / 2))
                    tmp = np.pad(tmp, [[pad_val, N - aper_size - pad_val], [pad_val, N - aper_size - pad_val]])
                    if tmp.shape[0] < D.shape[0]:
                        tmp = np.pad(tmp, [[0, 1], [0, 1]])
                    D[:, :, count] = tmp
                    count = count + 1
                else:
                    pad_val = int(np.round((N - aper_size) / 2))
                    tmp = Rmn * np.cos(m * phi) * mask_circ
                    tmp = np.pad(tmp, [[pad_val, N - aper_size - pad_val], [pad_val, N - aper_size - pad_val]])
                    if tmp.shape[0] < D.shape[0]:
                        tmp = np.pad(tmp, [[0, 1], [0, 1]])
                    D[:, :, count] = tmp
                    count = count + 1

                    tmp = Rmn * np.sin(m * phi) * mask_circ
                    tmp = np.pad(tmp, [[pad_val, N - aper_size - pad_val], [pad_val, N - aper_size - pad_val]])
                    if tmp.shape[0] < D.shape[0]:
                        tmp = np.pad(tmp, [[0, 1], [0, 1]])
                    D[:, :, count] = tmp
                    count = count + 1
    return D


def gaussian(N, sigma):
    """
    gaussian distribution generation
    :param N: size
    :param sigma: standard deviation. Its square is the variance.
    :return: normalized gaussian distribution, 2d ndarray
    """
    xl = np.linspace(-10, 10, N)
    x, y = np.meshgrid(xl, xl, indexing='xy')
    g = np.exp(-(x**2 + y**2)/(2*sigma**2))
    g = (g-np.min(g))/(np.max(g)-np.min(g))
    return g


def fresnel_zone_plate(f, wvl, D, N):
    """
    generate the binary fresnel zone plate according to focal length, wavelength, diameter and size
    reference: http://zoneplate.lbl.gov/theory
    :param f: focal length
    :param wvl: wave length
    :param D: diameter
    :param N: how many pixels are included
    :return: a matrix, ndarray
    """
    xl = np.linspace(-D/2, D/2, N)
    x, y = np.meshgrid(xl, xl, indexing='xy')
    r = np.sqrt(x**2 + y**2)
    fzp = np.zeros_like(r)
    n = 1
    while True:
        rn = np.sqrt(n*wvl*(f+n*wvl/4))
        if rn > D/2:
            break
        else:
            rn_ = np.sqrt((n-1)*wvl*(f+(n-1)*wvl/4))
            fzp[(r>=rn_) & (r<rn)] = 1.
            n = n+2
    return fzp


def circular_point_grid(H, W, s_c, s_r):
    """
    delete ???
    H hight, how many rows
    W width, how many columns
    return x and y coordinates of those points, 2*N ndarray

    """
    # s_c = H / 6  # spacing along the circumference
    # s_r = H / 6  # spacing along the radius
    n_c = 2  # number of circles
    xg, yg = [0], [0]
    for i in range(n_c):
        rr = s_r * (i + 1)  # radius of the current circle
        nn = int(2 * pi * rr / s_c)  # how many points along this circumference
        theta = 2 * pi / nn  # rotation angle
        rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])  # rotation matrix
        v = [[rr], [0]]
        for j in range(nn):
            v = rot @ v
            vv = v
            xg.append(vv[0, 0])
            yg.append(-vv[1, 0])
    xg = np.array(xg)[np.newaxis, :]
    yg = np.array(yg)[np.newaxis, :]
    xy = np.concatenate((xg, yg), axis=0)
    return xy


def stitched_image(im_stack, im_positions, H, W):
    # delete??
    n, R, C = im_stack.shape
    im = np.zeros((H, W))
    for i in range(n):
        xp = im_positions[i, 0] + W/2
        yp = im_positions[i, 1] + H/2
        im[int(yp-R/2): int(yp-R/2)+R, int(xp-C/2): int(xp-C/2)+C] = im_stack[i, :, :]
    return im


import torch
def power_norm(complex_amplitude, n_photons=1e6):
    # normalize complext amplitude according to the number of photons
    # complex_amplitude: 2D or 4D complex tensor
    intensity = torch.abs(complex_amplitude)**2
    if len(complex_amplitude.shape) == 4: # 4D
        intensity_sum = intensity.sum(-1).sum(-1).unsqueeze(2).unsqueeze(2)
    else:
        intensity_sum = intensity.sum()
    norm_factor = torch.sqrt(n_photons/intensity_sum)
    ca_norm = complex_amplitude*norm_factor
    return ca_norm


def plot_3D_dots(x, y, z):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, y, z, c=z, cmap='Greens')
    plt.show()


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
        # z = np.dot(np.c_[1., x, y, x * y, x ** 2, y ** 2, x * y ** 2, x ** 2 * y, x ** 3, y ** 3], C)
        return z
    elif order==4:
        A = np.c_[constant, xdata, ydata, xdata * ydata, xdata ** 2, ydata ** 2, xdata * ydata ** 2, xdata ** 2 * ydata,
                  xdata ** 3, ydata ** 3, xdata*ydata**3, xdata**2*ydata**2, xdata**3*ydata, xdata**4, ydata**4]
        C, _, _, _ = scipy.linalg.lstsq(A, zdata)
        z = np.dot(np.c_[np.ones(x.shape[0]), x, y, x * y, x ** 2, y ** 2, x * y ** 2, x ** 2 * y, x ** 3, y ** 3,
                   x*y**3, x**2*y**2, x**3*y, x**4, y**4], C)
        return z
    else:
        print('Order should be smaller than 4')


from scipy.optimize import curve_fit
def func(data, A, B, C, D, E, F, G, H, I, J):
    x, y, z = data.T
    return A*x**3+B*y**3+C*x*y**2+D*x**2*y+E*x**2+F*y**2+G*x*y+H*x+I*y+J

def distance_matrix(xys):
    """
    calculate distance matrix of a set of 2D points
    :param xys: ndarray, N*2, N points
    :return: dis_matix,
    """
    N = xys.shape[0]
    dis_matrix = np.zeros((N, N))
    for i in range(N):
        xy = xys[i:i+1, :]  # rank 2
        dis_matrix[:, i] = np.sqrt(np.sum((xys - xy)**2, axis=1))  # rank 1
    return dis_matrix





