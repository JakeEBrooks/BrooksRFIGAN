import os
import logging
log = logging.getLogger(__name__)
from itertools import combinations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats.mstats import winsorize

from casatools import ms, msmetadata

class MSHandler():
    def __init__(self):
        self.mstool = ms()

    def open(self, MSpath):
        self.done()
        if os.path.exists(MSpath):
            self.ref = os.path.realpath(MSpath)
            log.info('Found measurement set at {}'.format(self.ref))
        else:
            raise RuntimeError("Failed to find a Measurement Set at {} ...".format(MSpath))
        msmd = msmetadata()

        msmd.open(self.ref)
        self.antids = msmd.antennaids()
        self.antnames = msmd.antennanames()
        self.spws = msmd.datadescids()
        self.chanfreqs = np.array([])
        self.chanwidths = np.array([])
        for w in self.spws:
            self.chanfreqs = np.append(self.chanfreqs,msmd.chanfreqs(w))
            self.chanwidths = np.append(self.chanwidths,msmd.chanwidths(w))
        self.chans_per_spw = len(self.chanfreqs)//len(self.spws)
        self.fieldnames = msmd.fieldnames()
        self.fieldids = msmd.fieldsforname()
        msmd.done()

        self.antidpairs = np.array(list(combinations(self.antids,2)))
        self.baselineids = np.arange(0,len(self.antidpairs))
        self.baselinenames = np.array(['{} x {}'.format(self.antnames[x],self.antnames[y]) for x,y in self.antidpairs])
    
    def getBaselineImages(self, blids, fields=None, preserve_corr_ax=False, preserve_imag=False):
        if fields == None:
            field_ids = self.fieldids
        elif np.all([x in self.fieldnames for x in fields]):
            field_ids = self.fieldids[[x in fields for x in self.fieldnames]]
        elif np.all([x in self.fieldids for x in fields]):
            field_ids = fields
        else:
            raise RuntimeError('That is not a valid field input for getBaselineImages()\nAvailable fields are {}'.format(self.fieldnames))

        self.mstool.open(self.ref)
        self.mstool.select({'antenna1':blids[:,0], 'antenna2':blids[:,1], 'field_id':field_ids})
        dat = self.mstool.getdata(['data', 'antenna1', 'antenna2'])
        for idpair in blids:
            bldata = dat['data'][:,:,(dat['antenna1'] == idpair[0]) & (dat['antenna2'] == idpair[1])].T.reshape(-1,1024,4)
            if preserve_corr_ax:
                if np.all(idpair==blids[0]):
                    all_bldata = bldata[np.newaxis,:,:,:]
                else:
                    all_bldata = np.append(all_bldata,bldata[np.newaxis,:,:,:],axis=0)
            else:
                if np.all(idpair==blids[0]):
                    all_bldata = bldata
                else:
                    all_bldata = np.append(all_bldata,bldata,axis=2)
        if not preserve_corr_ax:
            all_bldata = np.moveaxis(all_bldata,-1,0)
        self.mstool.done()
        if preserve_imag:
            return all_bldata
        else:
            return all_bldata.real
    
    def getBaselineMasks(self, blids, fields=None, preserve_corr_ax=False):
        if fields == None:
            field_ids = self.fieldids
        elif np.all([x in self.fieldnames for x in fields]):
            field_ids = self.fieldids[[x in fields for x in self.fieldnames]]
        elif np.all([x in self.fieldids for x in fields]):
            field_ids = fields
        else:
            raise RuntimeError('That is not a valid field input for getBaselineMasks()\nAvailable fields are {}'.format(self.fieldnames))

        self.mstool.open(self.ref)
        self.mstool.select({'antenna1':blids[:,0], 'antenna2':blids[:,1], 'field_id':field_ids})
        dat = self.mstool.getdata(['flag', 'antenna1', 'antenna2'])
        for idpair in blids:
            log.info('Fetching {} x {}'.format(self.antnames[idpair[0]], self.antnames[idpair[1]]))
            bldata = dat['flag'][:,:,(dat['antenna1'] == idpair[0]) & (dat['antenna2'] == idpair[1])].T.reshape(-1,1024,4)
            if preserve_corr_ax:
                if np.all(idpair==blids[0]):
                    all_bldata = bldata[np.newaxis,:,:,:]
                else:
                    all_bldata = np.append(all_bldata,bldata[np.newaxis,:,:,:],axis=0)
            else:
                if np.all(idpair==blids[0]):
                    all_bldata = bldata
                else:
                    all_bldata = np.append(all_bldata,bldata,axis=2)
        if not preserve_corr_ax:
            all_bldata = np.moveaxis(all_bldata,-1,0)
        self.mstool.done()
        return all_bldata
    
    def getScans(self, fields=None):
        if fields == None:
            field_ids = self.fieldids
        elif np.all([x in self.fieldnames for x in fields]):
            field_ids = self.fieldids[[x in fields for x in self.fieldnames]]
        elif np.all([x in self.fieldids for x in fields]):
            field_ids = fields
        else:
            raise RuntimeError('That is not a valid field input for getScans()\nAvailable fields are {}'.format(self.fieldnames))

        self.mstool.open(self.ref)
        self.mstool.select({'antenna1':0, 'antenna2':1, 'data_desc_id':0, 'field_id':field_ids})
        scans = self.mstool.getdata(['scan_number'])['scan_number']
        self.mstool.done()
        return scans
    
    def getPrisonBars(self, quack_cols: int, major_chan_rowflags: int, minor_chan_rowflags: int, fields=None):
        channels = np.arange(len(self.chanfreqs))
        col_flags = np.array([])
        row_flags = np.array([])
        scans = self.getScans(fields)
        for col in np.arange(0,len(scans)):
            if scans[col-1] != scans[col]:
                col_flags = np.append(col_flags,np.arange(quack_cols)+col)
        row_flags = np.append(row_flags,channels[:major_chan_rowflags])
        row_flags = np.append(row_flags,channels[- major_chan_rowflags:])
        for n in np.arange(1,len(self.spws)):
            row_flags = np.append(row_flags,np.arange(n*self.chans_per_spw - minor_chan_rowflags, n*self.chans_per_spw + minor_chan_rowflags))
        row_flags.sort()
        col_flags.sort()
        return row_flags.astype(int), col_flags.astype(int) 

    def done(self):
        self.mstool.done()
        self.mstool.close()

def remove_surfs(im_data, sig_levels=[17,11,5], kernel_radius=64, kernel_sig=32):
    assert im_data.ndim == 3
    data = np.abs(np.copy(im_data))
    surfs_buff = np.empty(data.shape)
    for i, img in enumerate(data):
        log.info('Computing surface {}/{}'.format(i+1,data.shape[0]))
        for sig in sig_levels:
            surf = gaussian_filter(img, radius=kernel_radius, sigma=kernel_sig)
            residual = img - surf
            img = np.clip(img, None, np.mean(img)+sig*np.std(residual))
        surf = gaussian_filter(img, radius=kernel_radius, sigma=kernel_sig)
        surfs_buff[i,:,:] = surf
    return np.abs(im_data) - surfs_buff

def winsorize_images(images, limits):
    assert images.ndim == 3
    databuff = np.empty(images.shape)
    for i,image in enumerate(images):
        databuff[i,:,:] = winsorize(image,limits=limits)
    return databuff

def clip_images(images, sigma):
    assert images.ndim == 3
    databuff = np.empty(images.shape)
    for i,image in enumerate(images):
        databuff[i,:,:] = np.clip(image, None, np.mean(image)+sigma*np.std(image))
    return databuff

def normalize(x, high=1, low=0):
    if x.size > 1:
        return (x - np.min(x))/(np.max(x) - np.min(x))*(high - low)+low
    else:
        raise RuntimeError('Input to normalise has size <= 1, I can\'t normalise this!')

def pad_for_cutouts(images, cutout_size=(128,1024), **padkwargs):
    assert images.ndim == 3
    time_padded_size = 0
    while time_padded_size < images.shape[1]:
        time_padded_size += cutout_size[0]
    freq_padded_size = 0
    while freq_padded_size < images.shape[2]:
        freq_padded_size += cutout_size[1]
    
    time_pads = time_padded_size - images.shape[1]
    freq_pads = freq_padded_size - images.shape[2]
    pads = np.array([[time_pads//2, time_pads - time_pads//2],[freq_pads//2, freq_pads - freq_pads//2]])
    log.info('Padded {} to time axis, {} to frequency axis'.format(pads[0,:], pads[1,:]))

    return np.pad(images,((0,0),(pads[0,0],pads[0,1]),(pads[1,0],pads[1,1])), **padkwargs)

def make_cutouts(images, cutout_size=(128,1024)):
    assert images.ndim == 3
    assert images.shape[1] % cutout_size[0] == 0
    assert images.shape[2] % cutout_size[1] == 0
    cutouts = np.empty((0,cutout_size[0],cutout_size[1]))
    for im in np.arange(images.shape[0]):
        cuts = np.array(np.split(images[im,:,:], images.shape[1]//cutout_size[0]))
        cutouts = np.append(cutouts,cuts,axis=0)
    log.info('Produced {} cutouts from {} input images'.format(cutouts.shape[0], images.shape[0]))
    return cutouts