# FOV-dependent-DeepSTORM3D
add field dependence to enable large-FOV imaging 

1. Before training the field-dependent localization network, one needs to modify some parameters in the "parameter_setting_exp.py" according to specific context. This includes settings of image size, noise, FOV segmentaion, et al. 

1. Then "training_fd1.py" can be run to train a netwrok, which usually takes days of time in our case. Additionally, "training_fd0.py" which ignores field dependence, can be run for comparison. 

3. Finally, given image folder and the trained network, one can run "inference.py" implement localization frame by frame. The localization result is saved in a .csv file in the given image folder.

4. We resort to ThunderSTORM, a plug-in of ImageJ, to demonstrate of localization results. 

3D super-resolved image of mitochondria (depth range: 0-4 um):

![](./mitochondria_full.gif )

3D super-resolved image of microtubules (depth range: 0-4 um):

![](./microtubules1.gif )
