import os
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from scipy.ndimage import convolve1d, gaussian_filter

import preprocessing

def normalize(x, high=1, low=0):
    if x.size > 1:
        return (x - np.min(x))/(np.max(x) - np.min(x))*(high - low)+low
    else:
        raise RuntimeError('Input to normalise has size <= 1, I can\'t normalise this!')

msh = preprocessing.MSHandler()

msh.open('/home/jake6238/Work/lockmanhole/FLAGGING/LH_MAN_FLAGGED/CY5209_20180102_FLAGTEST_JEBrooks_Manual.ms')
images = msh.getBaselineImages(msh.antidpairs)
masks = msh.getBaselineMasks(msh.antidpairs)
pbrow, pbcol = msh.getPrisonBars(3,9,3)
msh.done()

msh.open('/home/jake6238/Work/lockmanhole/FLAGGING/LH_MAN_FLAGGED/Test_SEFD_L_20180101_F1.ms')
images2 = msh.getBaselineImages(msh.antidpairs)
masks2 = msh.getBaselineMasks(msh.antidpairs)
pbrow2, pbcol2 = msh.getPrisonBars(3,9,3)
msh.done()

#Remove images with 0 in them from the training data
images = np.delete(images, 51, 0) # 2017, Pi/Da, RR/LL, spw5
images = np.delete(images, 48, 0) # 2017, Pi/Da, RR/LL, spw5
images = np.delete(images, 2, 0) # 2017, Mk2/Kn, RL/LR, spw4
images = np.delete(images, 1, 0) # 2017, Mk2/Kn, RL/LR, spw4
images2 = np.delete(images2, 2, 0) # 2018, Mk2/Kn, RL/LR, spw4
images2 = np.delete(images2, 1, 0) # 2018, Mk2/Kn, RL/LR, spw4

masks = np.delete(masks, 51, 0) # 2017, Pi/Da, RR/LL, spw5
masks = np.delete(masks, 48, 0) # 2017, Pi/Da, RR/LL, spw5
masks = np.delete(masks, 2, 0) # 2017, Mk2/Kn, RL/LR, spw4
masks = np.delete(masks, 1, 0) # 2017, Mk2/Kn, RL/LR, spw4
masks2 = np.delete(masks2, 2, 0) # 2018, Mk2/Kn, RL/LR, spw4
masks2 = np.delete(masks2, 1, 0) # 2018, Mk2/Kn, RL/LR, spw4

images = np.delete(images,pbcol,1)
images = np.delete(images,pbrow,2)
masks = np.delete(masks,pbcol,1)
masks = np.delete(masks,pbrow,2)

images2 = np.delete(images2,pbcol2,1)
images2 = np.delete(images2,pbrow2,2)
masks2 = np.delete(masks2,pbcol2,1)
masks2 = np.delete(masks2,pbrow2,2)

# Remove the Iridium signal
# images = np.delete(images, range(685,710),2)
# masks = np.delete(masks, range(685,710),2)
# images2 = np.delete(images2, range(685,710),2)
# masks2 = np.delete(masks2, range(685,710),2)

# fig,ax = plt.subplots(nrows=2,ncols=1)
# ax[0].imshow(images[10])
# ax[1].imshow(masks[10])
# plt.show()

images = preprocessing.remove_surfs(images, sig_levels=[7,5,3])
images2 = preprocessing.remove_surfs(images2, sig_levels=[7,5,3])

images = preprocessing.winsorize_images(images,limits=(0,0.005))
images2 = preprocessing.winsorize_images(images2,limits=(0,0.005))

images = preprocessing.pad_for_cutouts(images, mode='reflect')
masks = preprocessing.pad_for_cutouts(masks, mode='reflect')
images2 = preprocessing.pad_for_cutouts(images2, mode='reflect')
masks2 = preprocessing.pad_for_cutouts(masks2, mode='reflect')

image_cuts = preprocessing.make_cutouts(images)
mask_cuts = preprocessing.make_cutouts(masks)
image_cuts2 = preprocessing.make_cutouts(images2)
mask_cuts2 = preprocessing.make_cutouts(masks2)

image_cuts = np.concatenate((image_cuts,image_cuts2),axis=0)
mask_cuts = np.concatenate((mask_cuts,mask_cuts2),axis=0)

np.save('test_images.npy',image_cuts)
np.save('test_masks.npy',mask_cuts)