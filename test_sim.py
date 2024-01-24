import numpy as np
from rfisim_funcs import *
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
import torch
from torch.utils.data import TensorDataset, DataLoader
from models import UNet, ConvNet

from preprocessing import make_cutouts
from aoflagsim import get_aoflags

rcParams['font.family'] = 'Times New Roman'

#MNRAS_figsize_2col = (8.2, 4.6125) # calculated by (A4 width - 0.1 inch, (A4 width - 0.1 inch)*(9/16))
MNRAS_figsize_2col = (8.2, 3.5)
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

generator = UNet(convfilters=64).to("cuda")
ckpt_tar_path = './checkpoints/ckpt_810.tar'
checkpoint = torch.load(ckpt_tar_path)
generator.load_state_dict(checkpoint['Gstate'])
generator.eval()
def evaluate(images,limit=0.7,batch_size=8):
    images = torch.Tensor(images[:,np.newaxis,:,:])
    dl = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=False)
    out = torch.empty((0,1,128,1024),dtype=torch.bool,device="cuda")
    for i,image in enumerate(dl):
        pred = generator(image[0].to("cuda"))
        out = torch.cat((out, torch.where(pred>limit,True,False)))
    return out[:,0,:,:].to("cpu").numpy()

def normalize(x, high=1, low=0):
    if x.size > 1:
        return (x - np.min(x))/(np.max(x) - np.min(x))*(high - low)+low
    else:
        raise RuntimeError('Input to normalise has size <= 1, I can\'t normalise this!')
def get_f1score(predictions,masks):
    tps = np.sum(np.logical_and(predictions,masks))
    fps = np.sum(np.logical_and(predictions,np.logical_not(masks)))
    fns = np.sum(np.logical_and(np.logical_not(predictions),masks))
    return (2*tps)/(2*tps + fps + fns)
def make_rfi_look_real(rfiims):
    power_fluctuation = rng.uniform(size=rfiims.shape)
    noise = rng.normal(loc=0,scale=0.1,size=rfiims.shape) #should be extremely tight because structured noise will have been mostly removed by surface fit
    rfiims *= power_fluctuation
    rfiims += noise
    for i,img in enumerate(rfiims):
        rfiims[i] = normalize(np.abs(img))
    return rfiims

strengths = []
dnrs = []
gan_tprs = []
gan_tnrs = []
gan_f1scores = []
aof_tprs = []
aof_tnrs = []
aof_f1scores = []

for test in range(0,500):
    print(test)
    rng = np.random.default_rng()
    rfistrength = 10**rng.uniform(0,2)
    rfiims, rfitruth = make_rfi_images(640,10,wb_p=0.1,rfi_strength=rfistrength,rfi_strength_sig=3)
    rfiims = np.moveaxis(rfiims,(2,1,0),(0,2,1))
    rfitruth = np.moveaxis(rfitruth,(2,1,0),(0,2,1))

    rfiims = make_rfi_look_real(rfiims)

    rfiims = make_cutouts(rfiims)

    # fig, ax = plt.subplots(ncols=2,nrows=1,figsize=MNRAS_figsize_1col,dpi=200)
    # ax[0].imshow(np.load('exampleimage.npy').T, interpolation='None', origin='lower', aspect='auto', cmap=cp)
    # ax[1].imshow(rfiims[2].T, interpolation='None', origin='lower', aspect='auto', cmap=cp)
    # for i in ax:
    #     i.axis("off")
    # fig.text(0.28,0.05,"(a)")
    # fig.text(0.71,0.05,"(b)")
    # fig.savefig('sim_comparison.png',bbox_inches='tight')

    rfitruth = make_cutouts(rfitruth)
    ganmasks = evaluate(rfiims)
    aomasks = get_aoflags(rfiims)

    gan_tprs.append(np.sum(np.logical_and(ganmasks,rfitruth))/np.sum(rfitruth))
    gan_tnrs.append(np.sum(np.logical_and(np.logical_not(ganmasks),np.logical_not(rfitruth)))/np.sum(np.logical_not(rfitruth)))
    gan_f1scores.append(get_f1score(ganmasks, rfitruth))

    aof_tprs.append(np.sum(np.logical_and(aomasks,rfitruth))/np.sum(rfitruth))
    aof_tnrs.append(np.sum(np.logical_and(np.logical_not(aomasks),np.logical_not(rfitruth)))/np.sum(np.logical_not(rfitruth)))
    aof_f1scores.append(get_f1score(aomasks, rfitruth))

    strengths.append(rfistrength)
    dnrs.append(np.mean(rfiims[rfitruth.astype(bool)])/np.mean(rfiims[np.logical_not(rfitruth.astype(bool))]))

print(np.mean(np.array(gan_f1scores)[np.array(strengths) < 10]))

fig, ax = plt.subplots(nrows=1,ncols=3,figsize=MNRAS_figsize_2col,dpi=200)
ax[0].scatter(strengths,gan_f1scores,marker='x',c='black',s=15)
ax[1].scatter(strengths,gan_tprs,marker='x',c='black',s=15)
ax[2].scatter(strengths,gan_tnrs,marker='x',c='black',label='GAN',s=15)
ax[0].scatter(strengths,aof_f1scores,marker='+',c='blue',s=15)
ax[1].scatter(strengths,aof_tprs,marker='+',c='blue',s=15)
ax[2].scatter(strengths,aof_tnrs,marker='+',c='blue',label='AOFlagger',s=15)

ax[1].set_xlabel('$\mu_{RFI}$')

ax[2].legend(loc='lower right')

ax[0].set_ylim(-0.01,1.01)
ax[1].set_ylim(-0.01,1.01)
ax[2].set_ylim(0.89,1.01)
ax[0].set_xlim(1,105)
ax[1].set_xlim(1,105)
ax[2].set_xlim(1,105)

ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[2].set_xscale('log')

fig.text(0.22,-0.1,"(a)")
fig.text(0.5,-0.1,"(b)")
fig.text(0.78,-0.1,"(c)")

fig.savefig('simscores.png',bbox_inches='tight')