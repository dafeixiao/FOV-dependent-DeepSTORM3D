
from parameter_setting_exp import parameter_setting_exp
from DeepSTORM3D.GenerateTrainingExamples import gen_data
from DeepSTORM3D.Training_Localization_Model import learn_localization_cnn
import torch

# specified training parameters
setup_params = parameter_setting_exp()

setup_params['fd_flag'] = False  # switch off the FOV dependence, for comparison
setup_params['training_data_path'] = setup_params['path_curr_dir'] + "/training_data/data_exp/"  # training data folder
setup_params['results_path'] = setup_params['path_curr_dir'] + "/training_results/results_exp/"  # result folder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
setup_params['device'] = device

# generate training data
gen_data(setup_params)  # will save updated setup_params in the training data folder

# # learn a localization cnn
learn_localization_cnn(setup_params)  # will save the updated setup_params in the result folder