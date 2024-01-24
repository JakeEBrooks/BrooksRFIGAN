import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s : %(message)s', datefmt='%d/%m/%Y %H:%M:%S',filename='{}.log'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),filemode='a')
log = logging.getLogger(__name__)

from models import UNet, ConvNet

# Approximate time per epoch: 3m

## Config
EPOCHS = 200
device = "cuda"
learning_rate = 1e-6
validation_split_frac = 0.05
enable_tensorboard = True
enable_checkpointing = True

continuing = True
ckpt_tar_path = './checkpoints/ckpt_610.tar'
##

def weights_init(m):
    """
    Initialises the weights of module m as described in the PyTorch DCGAN tutorial.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data,0.0,0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

def normalize(x, high=1, low=0):
    if x.size > 1:
        return (x - np.min(x))/(np.max(x) - np.min(x))*(high - low)+low
    else:
        raise RuntimeError('Input to normalise has size <= 1, I can\'t normalise this!')

def s_to_hms(total_seconds):
    """Convenience function to get N seconds in HMS format"""
    hours = total_seconds // 3600
    minutes = (total_seconds - hours*3600) // 60
    seconds = (total_seconds - hours*3600 - minutes*60)
    return '{0}h {1}m {2}s'.format(int(hours), int(minutes), int(seconds))

class GeneratorLoss(nn.Module):
    def __init__(self, lambda_val = 100):
        super().__init__()
        self.lambda_val = lambda_val
        self.bceloss = nn.BCEWithLogitsLoss()
        self.l1loss = nn.L1Loss()
    def forward(self, disc_fake_output, generated_masks, real_masks):
        return self.bceloss(disc_fake_output, torch.ones_like(disc_fake_output)) + self.lambda_val*self.l1loss(generated_masks, real_masks)

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bceloss = nn.BCEWithLogitsLoss()
    def forward(self, disc_real_output, disc_fake_output):
        return self.bceloss(disc_fake_output, torch.zeros_like(disc_fake_output)) + self.bceloss(disc_real_output, torch.ones_like(disc_real_output))

## Prepare the data
# load and prepare the images and masks
ims = np.load('./training_images.npy').astype(np.float32)
masks = np.load('./training_masks.npy')
masks = np.where(masks==True, 0.9, 0.1).astype(np.float32)
noise = np.random.normal(loc=0,scale=0.05,size=masks.shape)
masks += noise
masks = np.clip(np.abs(masks),0,1)
for i,img in enumerate(ims):
    ims[i] = normalize(np.abs(img))

numims = len(ims)
log.info('Found {} images'.format(numims))
train_ims = ims[int(validation_split_frac*numims):,:,:]
train_masks = masks[int(validation_split_frac*numims):,:,:]
val_ims = ims[:int(validation_split_frac*numims),:,:]
val_masks = masks[:int(validation_split_frac*numims),:,:]
numtrainims, numvalims = len(train_ims), len(val_ims)
val_tbtest_ims = torch.Tensor(val_ims[:5,:,:])[:,None,:,:].to(device)
val_tbtest_masks = torch.Tensor(val_masks[:5,:,:])[:,None,:,:].to(device)
log.info('Divided data into {} training samples and {} validation samples'.format(numtrainims, numvalims))
del ims
del masks

train_dataset = TensorDataset(torch.Tensor(train_ims[:,np.newaxis,:,:]), torch.Tensor(train_masks[:,np.newaxis,:,:]))
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)

val_dataset = TensorDataset(torch.Tensor(val_ims[:,np.newaxis,:,:]), torch.Tensor(val_masks[:,np.newaxis,:,:]))
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

#Initialise the models and move them to the gpu
generator = UNet(convfilters=64).to(device)
discriminator = ConvNet(convfilters=64).to(device)

#Initialise the weights as described in the PyTorch DCGAN tutorial
generator.apply(weights_init)
discriminator.apply(weights_init)

#Initialise optimisers for G and D
G_opt = optim.Adam(generator.parameters(), lr=learning_rate)
D_opt = optim.SGD(discriminator.parameters(), lr=learning_rate)

#Initialise the Loss functions for G and D
G_lossfn = GeneratorLoss()
D_lossfn = DiscriminatorLoss()

#Initialise an L1 Loss function for testing the MAE between fake and real masks
MAELoss = nn.L1Loss().to(device)

#Initialise the tensorboard writer
if enable_tensorboard:
    tbwriter = SummaryWriter('runs/run_{}'.format(datetime.now().strftime('%Y%m%d_%H%M%S')))

# Load the state of the desired checkpoint
start_epoch = 0
if continuing and ckpt_tar_path != '':
    checkpoint = torch.load(ckpt_tar_path)
    start_epoch = checkpoint['epoch']+1
    generator.load_state_dict(checkpoint['Gstate'])
    discriminator.load_state_dict(checkpoint['Dstate'])
    G_opt.load_state_dict(checkpoint['Goptim'])
    D_opt.load_state_dict(checkpoint['Doptim'])
end_epoch = start_epoch+EPOCHS

# TRAINING LOOP
log.info('---- Starting training ----')
train_starttime = time.time()
for epoch in range(start_epoch,end_epoch):
    epoch_starttime = time.time()
    log.info('-- Starting epoch {}/{}'.format(epoch+1,end_epoch))

    # Set both models to training mode
    generator.train(True)
    discriminator.train(True)
    for step, (im_batch, mask_batch) in enumerate(train_dataloader):
        log.info('- Step {}'.format(step))
        #Load a batch of images and masks and move them to the selected device
        im_batch = im_batch.to(device)
        mask_batch = mask_batch.to(device)

        #Train and update the discriminator
        discriminator.zero_grad()
        generated_masks = generator(im_batch)
        disc_real_preds = discriminator(mask_batch)
        disc_fake_preds = discriminator(generated_masks.detach())
        D_loss = D_lossfn(disc_real_preds, disc_fake_preds)
        D_loss.backward()
        D_opt.step()

        #Train and update the generator
        generator.zero_grad()
        disc_fake_preds = discriminator(generated_masks)
        G_loss = G_lossfn(disc_fake_preds, generated_masks, mask_batch)
        G_loss.backward()
        G_opt.step()

        #Print some metrics to track training progress
        log.info('G Loss: {}, D Loss: {}'.format(G_loss.item(), D_loss.item()))
        log.info('Avg Real Guesses: {} (1), Avg Fake Guesses: {} (0)'.format(torch.mean(disc_real_preds).item(), torch.mean(disc_fake_preds).item()))
    
    ## Perform a validation loop
    log.info('')
    log.info('-- Validating epoch {}/{}'.format(epoch+1,end_epoch))
    val_disc_running_loss = 0
    val_gen_running_loss = 0
    val_disc_running_fake_guesses = 0
    val_disc_running_real_guesses = 0
    val_running_mae = 0
    # Set both models to evaluation mode for the validation loop
    generator.train(False)
    discriminator.train(False)
    with torch.no_grad():
        for vstep, (val_im_batch, val_mask_batch) in enumerate(val_dataloader):
            val_im_batch = val_im_batch.to(device)
            val_mask_batch = val_mask_batch.to(device)

            val_generated_masks = generator(val_im_batch)
            val_disc_real_preds = discriminator(val_mask_batch)
            val_disc_fake_preds = discriminator(val_generated_masks)

            val_G_loss = G_lossfn(val_disc_fake_preds, val_generated_masks, val_mask_batch)
            val_D_loss = D_lossfn(val_disc_fake_preds, val_disc_real_preds)

            val_disc_running_fake_guesses += torch.sum(val_disc_fake_preds)
            val_disc_running_real_guesses += torch.sum(val_disc_real_preds)
            val_disc_running_loss += val_D_loss
            val_gen_running_loss += val_G_loss
            val_running_mae += MAELoss(val_generated_masks, val_mask_batch)
    log.info('G Validation Loss: {}, D Validation Loss: {}'.format(val_gen_running_loss, val_disc_running_loss))
    log.info('Avg Real Guesses: {} (1), Avg Fake Guesses: {} (0)'.format(val_disc_running_real_guesses/numvalims, val_disc_running_fake_guesses/numvalims))
    log.info('Generated Masks MAE: {}'.format(val_running_mae))
    log.info('')

    #Update tensorboard
    log.info('Updating Tensorboard...')
    if enable_tensorboard:
        tbwriter.add_scalar('G Validation Loss',val_gen_running_loss,epoch)
        tbwriter.add_scalar('D Validation Loss',val_disc_running_loss,epoch)
        tbwriter.add_scalar('D Validation Avg Real Guesses',val_disc_running_real_guesses/numvalims,epoch)
        tbwriter.add_scalar('D Validation Avg Fake Guesses',val_disc_running_fake_guesses/numvalims,epoch)
        tbwriter.add_scalar('Fake Mask Validation MAE',val_running_mae,epoch)
        with torch.no_grad():
            tbtest_fake_masks = generator(val_tbtest_ims)
        tbwriter.add_images('Fake Masks',torch.movedim(tbtest_fake_masks,2,3),epoch)
        tbwriter.add_images('Real Masks',torch.movedim(val_tbtest_masks,2,3),epoch)
        tbwriter.add_images('Input Data',torch.movedim(val_tbtest_ims,2,3),epoch)
        tbwriter.flush()
    
    if (epoch+1) % 5 == 0 and enable_checkpointing:
        log.info('Checkpointing...')
        torch.save({'epoch':epoch,
                    'Gstate':generator.state_dict(),
                    'Dstate':discriminator.state_dict(),
                    'Goptim':G_opt.state_dict(),
                    'Doptim':D_opt.state_dict()},
                    './checkpoints/ckpt_{}.tar'.format(epoch+1))

    log.info('Epoch finished in {}'.format(s_to_hms(time.time() - epoch_starttime)))
    if epoch+1 != end_epoch:
        log.info('ETA: {}'.format(s_to_hms((end_epoch-(epoch+1))*(time.time() - epoch_starttime))))
    log.info('')
if enable_tensorboard:
    tbwriter.close()

log.info('---- Finished training ----')
log.info('Time taken: {}'.format(s_to_hms(time.time() - train_starttime)))
# END OF TRAINING LOOP


