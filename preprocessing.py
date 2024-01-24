import os
import logging
log = logging.getLogger(__name__)
from itertools import combinations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats.mstats import winsorize

from casatools import ms, msmetadata

class MSHandler():
    """Interfaces with a Measurement Set to provide easy access to time-frequency images"""
    def __init__(self):
        """Instatiates a CASA MS tool"""
        self.mstool = ms()

    def open(self, MSpath):
        """
        Attach the MSHandler tool to a particular Measurement Set. Closes any previous connection to an MS

        :param MSpath: The filepath to the MS directory. For example: "path/to/myMS.ms"
        """
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
        """
        Returns an array containing time-frequency images for the specified baselines. By default returns an array of shape (*images*, *timesteps*, *channels*)

        :param blids: A 2d array of antenna id pairs. Use the antidpairs attribute for easy access to all baselines.
            Example input: [[0,1],[2,4]] for the baselines (antenna 0, antenna 1), and (antenna 2, antenna 4)
        :param fields: A list of strings identifying fields to be included in the output. The time axis of the output is organised as it appears in the MS. See the CASA `ms.timesort <https://casadocs.readthedocs.io/en/stable/api/tt/casatools.ms.html#casatools.ms.ms.timesort>`_ 
            method for arranging in time.
        :param preserve_corr_ax: An option to return the requested data with the polarisation axis preserved.
        :param preserve_imag: An option to retain the complex phase information for each visibility.
        :returns: *numpy.ndarray*
        """
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
        """
        Returns a boolean array containing the flags for the requested visibilities. A value of *True* indicates the visibility is flagged. Typically used in conjuction with getBaselineImages(). Inputs are the same as for getBaselineImages().
        """
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
        """Returns an array indicating the scan each timestep belongs to. Called from getPrisonBars()"""
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
        """
        Returns two *numpy.ndarray* containing the rows/columns (channels/timesteps) that are persistently flagged in every image. This allows easy removal of redundant information from the images.
        
        :param quack_cols: An integer specifying the number of timesteps that are flagged at the start of each scan.
        :param major_chan_rowflags: An integer specifying the number of channels that are flagged at the edges of the total bandwidth.
        :param minor_chan_rowflags: An integer specifying the number of channels that are flagged at the edges of each spectral window.
        :param fields: Same as getBaselineImages()
        :returns: *numpy.ndarray*, *numpy.ndarray*
        """
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
        """Closes the connection to the current MS. Called at the start of every open()"""
        self.mstool.done()
        self.mstool.close()

def remove_surfs(im_data, sig_levels=[17,11,5], kernel_radius=32, kernel_sig=32):
    """
    Performs a gaussian blur operation to remove the low-level signal from a time-frequency image. The process is iterative to get the best fit. It performs *len(sig_levels)* iterations and
    clips values that deviate from the mean of the residuals (image - surface) by the current iterations' sigma level.

    :param im_data: Input images of shape (num_images, timesteps, channels)
    :param sig_levels: The sigma level outside of which will be clipped in that iteration. Also indicates the number of iterations to perform.
          
        Example: sig_levels=[9,7,5] will fit a surface after 3 iterations. After the first residuals are calculated, values exceeding 9*numpy.std(residual) are clipped from the residuals.
        After the second iteration, the threshold is 7*numpy.std(residual), and so on.
    :param kernel_radius: The radius of gaussian kernel to use. As per `gaussian_filter <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html>`_
        the total size of the kernel will be 2*kernel_radius + 1
    :param kernel_sig: The spread of the gaussian kernel. Following section 3.1 of `Offringa et al. 2010 <https://ui.adsabs.harvard.edu/abs/2010MNRAS.405..155O/abstract>`_ a choice of kernel_sig=kernel_radius
        is a good starting point.
    :returns: numpy.abs(image) - surface
    """
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
    """
    Returns copies of the input *images* winsorized using *limits*

    :param images: Input images of shape (num_images, timesteps, channels)
    :param limits: A tuple of two values indicating the lower and higher thresholds for winsorization.
        
        Example: Passing (0.05,0.1) to limits would set the lowest 5% of values to the 5th percentile, and the top 10% of values to the 90th percentile
    :returns: Winsorized *images*
    """
    assert images.ndim == 3
    databuff = np.empty(images.shape)
    for i,image in enumerate(images):
        databuff[i,:,:] = winsorize(image,limits=limits)
    return databuff

def clip_images(images, sigma):
    """Returns a copy of *images* with values deviating from the mean by *sigma*numpy.std(image)* clipped"""
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
    """Pads the input images to ensure that an integer number of cutouts of *cutout_size* can be made. Extra keyword arguments are passed to *numpy.pad*"""
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
    """Returns all the data contained in images among cutouts of size *cutout_size*

    :returns: *numpy.ndarray* of shape (N, cutout_size[0], cutout_size[1])
    """
    assert images.ndim == 3
    assert images.shape[1] % cutout_size[0] == 0
    assert images.shape[2] % cutout_size[1] == 0
    cutouts = np.empty((0,cutout_size[0],cutout_size[1]))
    for im in np.arange(images.shape[0]):
        cuts = np.array(np.split(images[im,:,:], images.shape[1]//cutout_size[0]))
        cutouts = np.append(cutouts,cuts,axis=0)
    log.info('Produced {} cutouts from {} input images'.format(cutouts.shape[0], images.shape[0]))
    return cutouts