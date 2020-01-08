import unet3d.model.isensee2017 as mod


import nibabel as nib
#img = nib.load("/home/jbmai_sai/Documents/image.nii")
import numpy as np
image = nib.load("/home/jbmai_sai/Documents/image.nii")
image = image.get_data()


import matplotlib.pyplot as plt
def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = image[:, :,12]
slice_1 = image[:, :,14]
slice_2 = image[:, :, 16]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")  # doctest: +SKIP





#model = mod.isensee2017_model()

#model.summary()

#outputs = [layer.output_shape for layer in model.layers]
#final = outputs[-1]

#print(final)