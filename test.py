import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

from models import UNet, ConvNet

rcParams['font.family'] = 'Times New Roman'

MNRAS_figsize_2col = (8.2, 4.6125) # calculated by (A4 width - 0.1 inch, (A4 width - 0.1 inch)*(9/16))
MNRAS_figsize_1col = (4.1, 4.1) # If 1col width, plot is square with half width

title_fontsize = 16
axlabel_fontsize = 14
axtick_fontsize = 10

rcParams['axes.titlesize'] = title_fontsize
rcParams['axes.labelsize'] = axlabel_fontsize
rcParams['xtick.labelsize'] = axtick_fontsize
rcParams['ytick.labelsize'] = axtick_fontsize

cvals = np.ones((256, 4))
cvals[:,0] = np.append(np.linspace(20/256,0/256,128),np.linspace(0/256,220/256,128))
cvals[:,1] = np.append(np.linspace(20/256,114/256,128),np.linspace(114/256,220/256,128))
cvals[:,2] = np.append(np.linspace(20/256,220/256,128),np.linspace(220/256,0/256,128))
cp = ListedColormap(cvals)

binarycvals = np.ones((256, 4))
binarycvals[:,0] = np.linspace(20/256,240/256,256)
binarycvals[:,1] = np.linspace(20/256,40/256,256)
binarycvals[:,2] = np.linspace(40/256,40/256,256)
binarycp = ListedColormap(binarycvals)

def normalize(x, high=1, low=0):
    if x.size > 1:
        return (x - np.min(x))/(np.max(x) - np.min(x))*(high - low)+low
    else:
        raise RuntimeError('Input to normalise has size <= 1, I can\'t normalise this!')

def get_f1score(predictions,masks):
    precision = np.sum(np.logical_and(predictions,masks))/np.sum(predictions)
    recall = np.sum(np.logical_and(predictions,masks))/np.sum(masks)
    return 2/((1/recall)+(1/precision))
def get_tpr(predictions,masks):
    return np.sum(np.logical_and(predictions,masks))/np.sum(masks)
def get_tnr(predictions,masks):
    return np.sum(np.logical_and(np.logical_not(predictions),np.logical_not(masks)))/np.sum(np.logical_not(masks))

test_images = np.load('data/training_images.npy')
test_masks = np.load('data/training_masks.npy').astype(bool)
for i,img in enumerate(test_images):
    test_images[i] = normalize(np.abs(img))

print(np.mean(test_images[test_masks]))

generator = UNet(convfilters=64).to("cuda")
discriminator = ConvNet(convfilters=64).to("cuda")
ckpt_tar_path = './checkpoints/ckpt_810.tar'
checkpoint = torch.load(ckpt_tar_path)
generator.load_state_dict(checkpoint['Gstate'])
discriminator.load_state_dict(checkpoint['Dstate'])
generator.eval()
discriminator.eval()
def evaluate(images,limit=0.7,batch_size=8):
    images = torch.Tensor(images[:,np.newaxis,:,:])
    dl = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    out = torch.empty((0,1,128,1024),dtype=torch.bool,device="cuda")
    for i,image in enumerate(dl):
        pred = generator(image[0].to("cuda"))
        out = torch.cat((out, torch.where(pred>limit,True,False)))
    return out[:,0,:,:].to("cpu").numpy()

pred_masks = evaluate(test_images)

print("##### Stats")
print("TPR: ", get_tpr(pred_masks,test_masks)*100)
print("TNR: ", get_tnr(pred_masks,test_masks)*100)
print("F1 Score: ",get_f1score(pred_masks,test_masks)*100)

#remove pads
test_images = test_images[:,:,45:980] #check these numbers
test_masks = test_masks[:,:,45:980]
pred_masks = pred_masks[:,:,45:980]
ind = 2
fig, ax = plt.subplots(ncols=3,nrows=1,figsize=MNRAS_figsize_1col,dpi=200)
ax[0].imshow(test_images[ind].T, interpolation='None', origin='lower', aspect='auto', cmap=cp)
ax[1].imshow(test_masks[ind].T, interpolation='None', origin='lower', aspect='auto', cmap=binarycp)
ax[2].imshow(pred_masks[ind].T, interpolation='None', origin='lower', aspect='auto', cmap=binarycp)
for i in ax:
    i.axis("off")
fig.text(0.22,0.05,"(a)")
fig.text(0.5,0.05,"(b)")
fig.text(0.78,0.05,"(c)")
fig.savefig('example_output.png',bbox_inches='tight')