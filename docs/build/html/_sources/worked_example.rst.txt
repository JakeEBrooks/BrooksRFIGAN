Worked Example
==============

This section describes a full workflow from accessing a measurement set containing the 'ground truth' flags, all the way to training a model.

Acessing the MS
---------------
Unfortunately, CASA does not yet have an easy way of accessing the complete time-frequency information for each baseline. Since the base element in the MS format is 
an array of (channel number)x(polarisations), some work has to be done to get the necessary images out of the MS. The