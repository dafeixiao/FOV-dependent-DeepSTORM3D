# ======================================================================================================================
# train FOV-dependent DeepSTORM3D, with PPG3D (iN the folder PSFI3D) as the PSF generator

# ======================================================================================================================

# import the data generation and localization net learning functions

from parameter_setting_exp import parameter_setting_exp
from DeepSTORM3D.GenerateTrainingExamples import gen_data
from DeepSTORM3D.Training_Localization_Model import learn_localization_cnn
import torch


# specified training parameters
setup_params = parameter_setting_exp()  # by default, field-dependence is on.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set GPU
setup_params['device'] = device


# generate training data
gen_data(setup_params)  # will save updated setup_params in the training data folder

# # learn a localization cnn
learn_localization_cnn(setup_params)  # will save the updated setup_params in the result folder


