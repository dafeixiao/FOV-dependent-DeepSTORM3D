# after running training_fd1.py or training_fd0.py, the trained network and relevant files will be saved
# then this file can be run to localize images in a given folder

from DeepSTORM3D.Testing_Localization_Model import test_model
import os

path_curr = os.getcwd()
path_results = path_curr + '/training_results/results_exp/'  # the training result folder, change accordingly
postprocessing_params = {'thresh': 10, 'radius': 4}

path_exp_data = '/bigdata/Dafei_FDSTORM/tubules2_4/'  # image folder, change accordingly

xyz_rec, conf_rec = test_model(path_results, postprocessing_params, path_exp_data)
