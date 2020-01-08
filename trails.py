import os
import glob

config = dict()
config["image_shape"] = (128, 128, 128)  # This determines what shape the images will be cropped/resampled to.
config["patch_shape"] = None  # switch to None to train on the whole image
config["labels"] = (1, 2, 4)  # the label numbers on the input image
config["n_base_filters"] = 16
config["n_labels"] = len(config["labels"])
config["all_modalities"] = ["t1", "t1ce", "flair", "t2"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["nb_channels"] = len(config["training_modalities"])
if "patch_shape" in config and config["patch_shape"] is not None:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["patch_shape"]))
else:
    config["input_shape"] = tuple([config["nb_channels"]] + list(config["image_shape"]))
config["truth_channel"] = config["nb_channels"]
config["deconvolution"] = True  # if False, will use upsampling instead of deconvolution

config["batch_size"] = 1
config["validation_batch_size"] = 2
config["n_epochs"] = 500  # cutoff the training after this many epochs
config["patience"] = 10  # learning rate will be reduced after this many epochs if the validation loss is not improving
config["early_stop"] = 50  # training will be stopped after this many epochs without the validation loss improving
config["initial_learning_rate"] = 5e-4
config["learning_rate_drop"] = 0.5  # factor by which the learning rate will be reduced
config["validation_split"] = 0.8  # portion of the data that will be used for training
config["flip"] = False  # augments the data by randomly flipping an axis during
config["permute"] = True  # data shape must be a cube. Augments the data by permuting in various directions
config["distort"] = None  # switch to None if you want no distortion
config["augment"] = config["flip"] or config["distort"]
config["validation_patch_overlap"] = 0  # if > 0, during training, validation patches will be overlapping
config["training_patch_start_offset"] = (16, 16, 16)  # randomly offset the first patch index by up to this offset
config["skip_blank"] = True  # if True, then patches without any target will be skipped

config["data_file"] = os.path.abspath("brats_data.h5")
config["model_file"] = os.path.abspath("isensee_2017_model.h5")
config["training_file"] = os.path.abspath("isensee_training_ids.pkl")
config["validation_file"] = os.path.abspath("isensee_validation_ids.pkl")
config["overwrite"] = False  # If True, will previous files. If False, will use previously written files.

from nilearn.image import reorder_img, new_img_like

#from .nilearn_custom_utils.nilearn_utils import crop_img_to
data_shape = tuple([0, 3] + list(config['image_shape']))

#print(data_shape)

import nibabel as nib
img = nib.load("/home/jbmai_sai/Documents/image.nii")
import numpy as np
image = nib.load("/home/jbmai_sai/Documents/image.nii")
#import numpy as np

# Get data from nibabel image object (returns numpy memmap object)
#img_data = img.get_data()

interpolation  ="linear"
image = reorder_img(image, resample=interpolation)
#print(image.shape)
zoom_level = np.divide(config["image_shape"], image.shape)

print(image.header.get_zooms())
new_spacing = np.divide(image.header.get_zooms(), zoom_level)
print(new_spacing)
#new_data = resample_to_spacing(image.get_data(), image.header.get_zooms(), new_spacing,
          #                     interpolation=interpolation)
#new_affine = np.copy(image.affine)
#np.fill_diagonal(new_affine, new_spacing.tolist() + [1])
#new_affine[:3, 3] += calculate_origin_offset(new_spacing, image.header.get_zooms())


exit()

def fetch_training_data_files(return_subject_ids=False):
    training_data_files = list()
    subject_ids = list()
    for subject_dir in glob.glob(os.path.join(os.path.dirname('/home/jbmai_sai/Downloads/'), "Pre-operative_TCGA_LGG_NIfTI_and_Segmentations", "*", "*")):
        print(subject_dir)
        x = os.path.basename(subject_dir)
        print(x)
        subject_ids.append(x)
        subject_files = list()
        for modality in config["training_modalities"] + ["truth"]:
            subject_files.append(os.path.join(subject_dir, modality + ".nii.gz"))
        training_data_files.append(tuple(subject_files))
    if return_subject_ids:
        return training_data_files, subject_ids
    else:
        return training_data_files



#import nibabel as nib
#img = nib.load("/home/jbmai_sai/Documents/image.nii")
#import numpy as np

# Get data from nibabel image object (returns numpy memmap object)
#img_data = img.get_data()

#print(type(img_data))
# Convert to numpy ndarray (dtype: uint16)
#img_data_arr = np.asarray(img_data)

#print(img_data_arr.shape)

#from brats.train_isensee2017 import fetch_training_data_files

training_files, subject_ids = fetch_training_data_files(return_subject_ids=True)

print(training_files[0:5])