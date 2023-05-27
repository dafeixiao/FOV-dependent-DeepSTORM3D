# Import modules and libraries
import torch
from torch.utils.data import DataLoader
import csv
import pickle
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import glob
from skimage.io import imread
import time
import argparse
from DeepSTORM3D.data_utils import generate_batch, complex_to_tensor, ExpDataset, sort_names_tif, ExpDataset_
from DeepSTORM3D.cnn_utils import LocalizationCNN, LocalizationCNN_, LocalizationCNN_2
from DeepSTORM3D.vis_utils import ShowMaskPSF, ShowRecovery3D, ShowLossJaccardAtEndOfEpoch
from DeepSTORM3D.vis_utils import PhysicalLayerVisualization, ShowRecNetInput
from DeepSTORM3D.physics_utils import EmittersToPhases, NoiseLayer
from DeepSTORM3D.postprocess_utils import Postprocess
from DeepSTORM3D.assessment_utils import calc_jaccard_rmse
from DeepSTORM3D.helper_utils import normalize_01, xyz_to_nm
from helper_utils_ import show_imset, noise_processing
import scipy.io as sio
from skimage import io
from helper_utils_ import plot_3D_dots
from helper_utils_ import show_imset, noise_processing
from PSFI3D.PSF_interpolation import FDPSFGenerator0
from PSFI3D.PSF_interpolation import image_generation
from scipy.optimize import linear_sum_assignment

def dis_matrix(location1, location2):
    """
    :param location1: prediction
    :param location2: ground truth
    :return: matrix
    """
    dis = np.zeros((location1.shape[0], location2.shape[0]))
    for i in range(location1.shape[0]):
        for j in range(location2.shape[0]):
            dis[i, j] = np.sqrt(np.sum((location1[i, :]-location2[j, :])**2))
    return dis



def test_model(path_results, postprocess_params, exp_imgs_path=None, seed=66):

    # close all existing plots
    plt.close("all")

    # load assumed setup parameters
    path_params_pickle = path_results + 'setup_params.pickle'
    with open(path_params_pickle, 'rb') as handle:
        setup_params = pickle.load(handle)

    # run on GPU if available
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    setup_params['device'] = device
    torch.backends.cudnn.benchmark = True

    # load learning results
    path_learning_pickle = path_results + 'learning_results.pickle'
    with open(path_learning_pickle, 'rb') as handle:
        learning_results = pickle.load(handle)

    # plot metrics evolution in training for debugging
    # plt.figure()
    # ShowLossJaccardAtEndOfEpoch(learning_results, learning_results['epoch_converged'])


    # build model and convert all the weight tensors to GPU is available
    if setup_params['fd_flag']:
        # cnn = LocalizationCNN_(setup_params)
        cnn = LocalizationCNN_2(setup_params)
    else:
        cnn = LocalizationCNN(setup_params)
    cnn.to(device)

    # load learned weights
    cnn.load_state_dict(torch.load(path_results + 'weights_best_loss.pkl', map_location=device))

    # post-processing module on CPU/GPU
    thresh, radius = postprocess_params['thresh'], postprocess_params['radius']
    postprocessing_module = Postprocess(thresh, radius, setup_params)

    # if no experimental imgs are supplied then sample a random example
    if exp_imgs_path is None:
        # ==============================================================================================================
        # generate a simulated test image
        # ==============================================================================================================

        # set random number generators given the seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        setup_params_copy = setup_params.copy()

        total_W = 161 + (6 - 1) * (161 - 2 * 20)
        setup_params_copy['H'], setup_params_copy['W'] = total_W, total_W
        # setup_params_copy['H'], setup_params_copy['W'] = setup_params_copy['total_W'], setup_params_copy['total_W']

        setup_params_copy['nsig_unif'] = True
        setup_params_copy['num_particles_range'] = [240, 240 + 1]
        xyz, Nphotons = generate_batch(1, setup_params_copy)

        my_shift = total_W*0.13/2 - 161/2*0.13
        xyz[:, :, 0] = xyz[:, :, 0] - my_shift
        xyz[:, :, 1] = xyz[:, :, 1] + my_shift

        # # set the same z to form a plane
        # xyz[:, :, 2] = 0.5
        # Nphotons = np.ones_like(Nphotons)*2e4

        # a circle
        # theta = np.linspace(0, 2*np.pi, 10)
        # r = 1  # um
        # x, y = r*np.sin(theta), r*np.cos(theta)
        # z = np.ones_like(x)*(-1.0)
        # xyz = np.c_[x, y, z]
        # xyz = xyz[np.newaxis, :, :]
        # Nphotons = np.ones((1, xyz.shape[1])) * 1e4

        xyzps = np.c_[xyz[0, :, :], Nphotons[0, :]]
        image0 = image_generation(xyzps, setup_params['total_W'], setup_params)

        setup_params_copy['H'], setup_params_copy['W'] = setup_params_copy['total_W'], setup_params_copy['total_W']
        # setup_params_copy['read_noise_baseline_range'] = [3, 6]
        # setup_params_copy['read_noise_std_range'] = [3, 6]

        noise_layer = NoiseLayer(setup_params_copy)  # add noise
        im_4D_tensor = torch.tensor(image0, device=setup_params['device']).unsqueeze(0).unsqueeze(0)
        im_noisy = noise_layer(im_4D_tensor)  # background, poisson, read noise
        image0 = im_noisy[0, 0, :, :].cpu().numpy()

        # show_imset(image0)

        cropped_imxys = torch.zeros(1, setup_params['xy_centers_num'], setup_params['H'], setup_params['W'])
        for i in range(setup_params['xy_centers_num']):
            (r, c) = setup_params['crop_rcs'][i]
            sub_image = image0[r:r + setup_params['H'], c:c + setup_params['W']]
            # normalize image according to the training setting
            if setup_params['project_01'] is True:
                sub_image = normalize_01(sub_image)
            else:
                sub_image = (sub_image - setup_params['global_factors'][0]) / setup_params['global_factors'][1]
            # alter the mean and std to match the training set
            if setup_params['project_01'] is True:
                sub_image = (sub_image - sub_image.mean()) / (sub_image.std()+1e-12)
                sub_image = sub_image * setup_params['train_stats'][1] + setup_params['train_stats'][0]

            cropped_imxys[0, i, :, :] = torch.from_numpy(sub_image)

        exp_im_tensor = cropped_imxys
        # valid_xy_half = (setup_params['H'] - 2 * setup_params['clear_dist']) / 2 * setup_params['pixel_size_FOV']
        valid_xy_half = (setup_params['H'] - 2 * setup_params['clear_dist']) / 2 * setup_params['pixel_size_FOV']

        with torch.set_grad_enabled(False):
            xyz_rec = None
            conf_rec = None
            for i, (xc, yc) in enumerate(setup_params['xy_centers']):
                exp_sub_im_tensor = exp_im_tensor[:, i:i + 1, :, :]
                show_sub_im = exp_sub_im_tensor[0, 0, :, :].numpy()
                if setup_params['fd_flag']:
                    xy_map = setup_params['xy_maps'][i].astype(np.float32)
                    xy_map = torch.from_numpy(xy_map[np.newaxis, :, :, :], )
                    exp_sub_im_tensor = torch.concat((exp_sub_im_tensor, xy_map), dim=1)

                exp_sub_im_tensor = exp_sub_im_tensor.to(device)
                pred_volume = cnn(exp_sub_im_tensor)
                sub_xyz_rec, sub_conf_rec = postprocessing_module(pred_volume)

                # plt.figure(111)
                # plt.imshow(show_sub_im, cmap='gray')

                # filter and concatenate, according to valid_xy_half
                if sub_xyz_rec is not None:
                    xx, yy = sub_xyz_rec[:, 0], sub_xyz_rec[:, 1]

                    # show_xy = sub_xyz_rec[:, 0:2]
                    # show_xy_ = show_xy / 0.13 + 80
                    # plt.plot(show_xy_[:, 0], show_xy_[:, 1], 'r*')

                    tfx = (xx >= -valid_xy_half) & (xx <= valid_xy_half)
                    tfy = (yy >= -valid_xy_half) & (yy <= valid_xy_half)
                    valid_indices = np.where(tfx & tfy)[0]
                    if valid_indices.shape[0] != 0:
                        sub_xyz_rec = sub_xyz_rec[valid_indices, :]
                        sub_conf_rec = sub_conf_rec[valid_indices]

                        sub_xyz_rec[:, 0] = sub_xyz_rec[:, 0] + xc + 0.0625  # offset
                        sub_xyz_rec[:, 1] = sub_xyz_rec[:, 1] + yc + 0.0625
                        if xyz_rec is None:
                            xyz_rec = sub_xyz_rec
                            conf_rec = sub_conf_rec
                        else:
                            xyz_rec = np.concatenate((xyz_rec, sub_xyz_rec), axis=0)
                            conf_rec = np.concatenate((conf_rec, sub_conf_rec), axis=0)
                # show
                # plt.show()


        xyz_gt = np.squeeze(xyz, 0)
        # if xyz_rec is not None:
        #     plt.figure()
        #     ShowRecovery3D(xyz_gt, xyz_rec)
        #     plt.title(f'gt num: {xyz_gt.shape[0]}, recovery num: {xyz_rec.shape[0]}')
        #     plt.show()

        threshold = 0.30
        dis = dis_matrix(xyz_rec, xyz_gt)
        row_ind, col_ind = linear_sum_assignment(dis)
        xyz_gt_matched = xyz_gt[col_ind]
        dis_rmse = np.sqrt(np.sum((xyz_rec[row_ind]-xyz_gt_matched)**2, axis=1))

        xyz_gt_ = xyz_gt_matched[dis_rmse < threshold, :]
        xyz_rec_ = xyz_rec[dis_rmse < threshold, :]
        dis_rmse_ = dis_rmse[dis_rmse < threshold]

        mean_lateral_rmse = np.mean(np.sqrt(np.sum((xyz_rec_[:, 0:2]-xyz_gt_[:, 0:2])**2, axis=1)))
        mean_axial_rmse = np.mean(np.abs((xyz_rec_[:, 2]-xyz_gt_[:, 2])))
        mean_rmse = np.mean(dis_rmse_)

        # print(f'matched ratio: {dis_rmse_.shape[0]/dis_rmse.shape[0]}, {mean_lateral_rmse}, {mean_axial_rmse}, {mean_rmse}')
        print(f'{dis_rmse_.shape[0]}, {mean_lateral_rmse}, {mean_axial_rmse}, {mean_rmse}')

        # print('end!')

        return xyz_rec, conf_rec, mean_lateral_rmse, mean_axial_rmse, dis_rmse_.shape[0]

    else:
        # read all imgs in the experimental data directory assuming ".tif" extension
        img_names = glob.glob(exp_imgs_path + '*.tif')
        # img_names = sort_names_tif(img_names)

        # if given only 1 image then show xyz in 3D and recovered image
        if len(img_names) == 1:

            # ==========================================================================================================
            # read experimental image and normalize it
            # ==========================================================================================================
            if setup_params['fd_flag']:
                exp_test_set = ExpDataset(img_names, setup_params)
            else:
                exp_test_set = ExpDataset(img_names, setup_params)

            exp_generator = DataLoader(exp_test_set, batch_size=1, shuffle=False)
            exp_im_tensor, full_image = next(iter(exp_generator))

            # ==========================================================================================================
            # predict the positions by post-processing the net's output
            # ==========================================================================================================

            # prediction using model
            cnn.eval()
            valid_xy_half = (setup_params['H'] - 2 * setup_params['clear_dist']) / 2 * setup_params['pixel_size_FOV']

            time_start = time.time()

            # if setup_params['fd_flag']:
            #     all_xy_maps = np.array(setup_params['xy_maps']).astype(np.float32)
            #     all_xy_maps = torch.from_numpy(all_xy_maps)
            #     exp_all_im_tensor = torch.concat((torch.unsqueeze(exp_im_tensor[0, :, :, :], 1), all_xy_maps), dim=1)
            # exp_all_im_tensor = exp_all_im_tensor.to(device)
            # exp_all_im_tensor = exp_all_im_tensor[:12, :, :, :]
            # pred_volume = cnn(exp_all_im_tensor)
            # print('done')
            # exit()

            with torch.set_grad_enabled(False):
                xyz_rec = None
                conf_rec = None

                for i, (xc, yc) in enumerate(setup_params['xy_centers']):
                    exp_sub_im_tensor = exp_im_tensor[:, i:i + 1, :, :]

                    show_sub_im = exp_sub_im_tensor[0, 0, :, :].numpy()

                    if setup_params['fd_flag']:
                        xy_map = setup_params['xy_maps'][i].astype(np.float32)
                        xy_map = torch.from_numpy(xy_map[np.newaxis, :, :, :], )
                        exp_sub_im_tensor = torch.concat((exp_sub_im_tensor, xy_map), dim=1)

                    exp_sub_im_tensor = exp_sub_im_tensor.to(device)
                    pred_volume = cnn(exp_sub_im_tensor)
                    sub_xyz_rec, sub_conf_rec = postprocessing_module(pred_volume)

                    # filter and concatenate, according to valid_xy_half
                    if sub_xyz_rec is not None:
                        xx, yy = sub_xyz_rec[:, 0], sub_xyz_rec[:, 1]
                        tfx = (xx >= -valid_xy_half) & (xx <= valid_xy_half)
                        tfy = (yy > -valid_xy_half) & (yy <= valid_xy_half)
                        valid_indices = np.where(tfx & tfy)[0]
                        if valid_indices.shape[0] != 0:
                            sub_xyz_rec = sub_xyz_rec[valid_indices, :]
                            sub_conf_rec = sub_conf_rec[valid_indices]

                            # show sub_xy
                            # plt.figure(111)
                            # plt.imshow(show_sub_im, cmap='gray')
                            # show_xy = sub_xyz_rec[:, 0:2]
                            # show_xy_ = show_xy / 0.13 + 80
                            # plt.plot(show_xy_[:, 0], show_xy_[:, 1], 'r*')
                            # plt.show()

                            sub_xyz_rec[:, 0] = sub_xyz_rec[:, 0] + xc
                            sub_xyz_rec[:, 1] = sub_xyz_rec[:, 1] + yc

                            if xyz_rec is None:
                                xyz_rec = sub_xyz_rec
                                conf_rec = sub_conf_rec
                            else:
                                xyz_rec = np.concatenate((xyz_rec, sub_xyz_rec), axis=0)
                                conf_rec = np.concatenate((conf_rec, sub_conf_rec), axis=0)

            time_end = time.time()
            print(f'detect {xyz_rec.shape[0]} emitters, time: {time_end-time_start} s')
            # show input image
            H, W, pixel_size_FOV = full_image.shape[1], full_image.shape[2], setup_params['pixel_size_FOV']
            ch, cw = np.floor(H / 2), np.floor(W / 2)
            fig100 = plt.figure(1012)
            im_np = np.squeeze(full_image.numpy()[0, :, :])
            imfig = plt.imshow(im_np, cmap='gray')
            plt.plot(xyz_rec[:, 0] / pixel_size_FOV + cw, xyz_rec[:, 1] / pixel_size_FOV + ch, 'r+')
            plt.title(f'Num of emitters: {xyz_rec.shape[0]}')
            fig100.colorbar(imfig)
            plt.show()
            #
            # # plot recovered 3D positions compared to GT
            plt.figure(1013)
            ax = plt.axes(projection='3d')
            ax.scatter(xyz_rec[:, 0], xyz_rec[:, 1], xyz_rec[:, 2], c='r', marker='^', label='DL', depthshade=False)
            ax.set_xlabel('X [um]')
            ax.set_ylabel('Y [um]')
            ax.set_zlabel('Z [um]')
            plt.title('3D Recovered Positions')
            plt.show()

            # show the overlay image

            xyzps = np.c_[xyz_rec, 1e4*np.ones((xyz_rec.shape[0], 1))]
            im_pred = image_generation(xyzps, full_image.shape[1], setup_params)
            im_pred = (im_pred-im_pred.min())/(im_pred.max()-im_pred.min())

            im_np = (im_np-im_np.min())/(im_np.max()-im_np.min())
            fig102 = plt.figure(102)
            im_pred_3 = np.stack((im_np,) * 3, axis=-1)
            im_pred_3[:, :, [0, 2]] = 0  # get the color
            mask = im_pred > 0.08
            mask = np.expand_dims(mask, axis=2)
            im102 = np.stack((im_np,) * 3, axis=-1)
            im102 = im102 * (1 - mask) + im_pred_3 * mask
            plt.imshow(im102)
            plt.savefig('overlay.png', dpi=600, bbox_inches="tight")
            plt.show()


            return xyz_rec, conf_rec

        else:

            # ==========================================================================================================
            # create a data generator to efficiently load imgs for temporal acquisitions
            # ==========================================================================================================

            # instantiate the data class and create a data loader for testing
            num_imgs = len(img_names)

            if setup_params['fd_flag']:
                # exp_test_set = ExpDataset_(img_names, setup_params)
                exp_test_set = ExpDataset(img_names, setup_params)
            else:
                exp_test_set = ExpDataset(img_names, setup_params)

            exp_generator = DataLoader(exp_test_set, batch_size=1, shuffle=False)

            # time the entire dataset analysis
            tall_start = time.time()

            # needed pixel-size for plotting if only few images are in the folder
            visualize_flag, pixel_size_FOV = num_imgs < 100, setup_params['pixel_size_FOV']

            # needed recovery pixel size and minimal axial height for turning ums to nms
            psize_rec_xy, zmin = setup_params['pixel_size_rec'], setup_params['zmin']

            # process all experimental images
            cnn.eval()
            results = np.array(['frame', 'x [nm]', 'y [nm]', 'z [nm]', 'intensity [au]'])

            valid_xy_half = (setup_params['H'] - 2 * setup_params['clear_dist']) / 2 * setup_params['pixel_size_FOV']

            with torch.set_grad_enabled(False):
                for im_ind, (exp_im_tensor, full_image) in enumerate(exp_generator):
                    # print current image number
                    print('Processing Image [%d/%d]' % (im_ind + 1, num_imgs))
                    # time each frame
                    tfrm_start = time.time()

                    xyz_rec = None
                    conf_rec = None
                    for i, (xc, yc) in enumerate(setup_params['xy_centers']):
                        exp_sub_im_tensor = exp_im_tensor[:, i:i+1, :, :]

                        # show_sub_im = exp_sub_im_tensor[0, 0, :, :].numpy()

                        if setup_params['fd_flag']:
                            xy_map = setup_params['xy_maps'][i].astype(np.float32)
                            xy_map = torch.from_numpy(xy_map[np.newaxis, :, :, :], )
                            exp_sub_im_tensor = torch.concat((exp_sub_im_tensor, xy_map), dim=1)

                        exp_sub_im_tensor = exp_sub_im_tensor.to(device)
                        pred_volume = cnn(exp_sub_im_tensor)
                        sub_xyz_rec, sub_conf_rec = postprocessing_module(pred_volume)

                        # filter and concatenate, according to valid_xy_half
                        if sub_xyz_rec is not None:
                            xx, yy = sub_xyz_rec[:, 0], sub_xyz_rec[:, 1]
                            tfx = (xx >= -valid_xy_half) & (xx <= valid_xy_half)
                            tfy = (yy > -valid_xy_half) & (yy <= valid_xy_half)
                            valid_indices = np.where(tfx & tfy)[0]
                            if valid_indices.shape[0] != 0:
                                sub_xyz_rec = sub_xyz_rec[valid_indices, :]
                                sub_conf_rec = sub_conf_rec[valid_indices]

                                # show sub_xy
                                # plt.figure(111)
                                # plt.imshow(show_sub_im, cmap='gray')
                                # show_xy = sub_xyz_rec[:, 0:2]
                                # show_xy_ = show_xy / 0.13 + 80
                                # plt.plot(show_xy_[:, 0], show_xy_[:, 1], 'r*')
                                # plt.show()

                                sub_xyz_rec[:, 0] = sub_xyz_rec[:, 0] + xc
                                sub_xyz_rec[:, 1] = sub_xyz_rec[:, 1] + yc
                                if xyz_rec is None:
                                    xyz_rec = sub_xyz_rec
                                    conf_rec = sub_conf_rec
                                else:
                                    xyz_rec = np.concatenate((xyz_rec, sub_xyz_rec), axis=0)
                                    conf_rec = np.concatenate((conf_rec, sub_conf_rec), axis=0)

                    # time it takes to analyze a single frame
                    tfrm_end = time.time() - tfrm_start

                    # if this is the first image, get the dimensions and the relevant center for plotting
                    if im_ind == 0:
                        N, H, W = full_image.size()
                        ch, cw = np.floor(H / 2), np.floor(W / 2)

                    # if prediction is empty then set number fo found emitters to 0
                    # otherwise generate the frame column and append results for saving
                    if xyz_rec is None:
                        nemitters = 0
                    else:
                        nemitters = xyz_rec.shape[0]
                        frm_rec = (im_ind + 1)*np.ones(nemitters)
                        xyz_save = xyz_to_nm(xyz_rec, H*2, W*2, psize_rec_xy, zmin)
                        results = np.vstack((results, np.column_stack((frm_rec, xyz_save, conf_rec))))

                    # if the number of imgs is small then plot each image in the loop with localizations
                    visualize_flag = False
                    if visualize_flag:
                        # show input image
                        fig100 = plt.figure(100)
                        im_np = np.squeeze(full_image.numpy()[0, :, :])
                        imfig = plt.imshow(im_np, cmap='gray')
                        plt.plot(xyz_rec[:, 0] / pixel_size_FOV + cw, xyz_rec[:, 1] / pixel_size_FOV + ch, 'r+')
                        plt.title('Single frame complete in {:.2f}s, found {:d} emitters'.format(tfrm_end, nemitters))
                        fig100.colorbar(imfig)
                        plt.show()
                        # plt.draw()
                        # plt.pause(0.05)
                        # plt.clf()

                    print('Single frame complete in {:.6f}s, found {:d} emitters'.format(tfrm_end, nemitters))
                    if im_ind == 0:
                        row_list = results.tolist()
                        with open(exp_imgs_path + str(setup_params['fd_flag'])+'_'+'localizations.csv', 'w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerows(row_list)
                        print(f'Results can be saved in {exp_imgs_path}.')



            # print the time it took for the entire analysis
            tall_end = time.time() - tall_start
            print('=' * 50)
            print('Analysis complete in {:.0f}h {:.0f}m {:.0f}s'.format(
                tall_end // 3600, np.floor((tall_end / 3600 - tall_end // 3600) * 60), tall_end % 60))
            print('=' * 50)

            # write the results to a csv file named "localizations.csv" under the exp img folder
            row_list = results.tolist()
            with open(exp_imgs_path + str(setup_params['fd_flag'])+'_'+'localizations.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(row_list)
                print(f'Results are saved in {exp_imgs_path}.')

            # return the localization results for the last image
            return xyz_rec, conf_rec


if __name__ == '__main__':

    # start a parser
    parser = argparse.ArgumentParser()

    # previously trained model
    parser.add_argument('--path_results', help='path to the results folder for the pre-trained model', required=True)

    # previously trained model
    parser.add_argument('--postprocessing_params', help='post-processing dictionary parameters', required=True)

    # path to the experimental images
    parser.add_argument('--exp_imgs_path', default=None, help='path to the experimental test images')

    # seed to run model
    parser.add_argument('--seed', default=66, help='seed for random test data generation')

    # parse the input arguments
    args = parser.parse_args()

    # run the data generation process
    xyz_rec, conf_rec = test_model(args.path_results, args.postprocessing_params, args.exp_imgs_path, args.seed)
