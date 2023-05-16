Usage
=====

Installation
------------

The library is written in pure Python, so it should be installable with a simple:

.. code-block:: console

    $ pip install /path/to/wheel

Which will install this package along with its dependencies. The version of tensorflow it installs is likely not optimized by default, though it will work. If you are planning on doing
big data experiments (rather than just playing around) I highly recommend `installing tensorflow from source <https://www.tensorflow.org/install/source>`_ for the best possible 
performance on either a CPU or a GPU.

Minimum Usage
-------------

The library comes with preprocessing functionailty described in `Brooks & Argo XXXX <hyperlink>`_, but this functionality is generally built to work with e-MERLIN.
The way e-MERLIN data is stored in a Measurement Set may differ from other telescopes, so it is generally safer to build your own input pipelines.

The core functionaility of the library revolves around the TRAINFUNCTION, which accepts a tensorflow dataset as input and produces both a trained generator model and a trained discriminator model.
By default, it will train these models using the parameters described in the paper, but there is an option to tweak these parameters, or even to provide your own Keras model. The tensorflow dataset format
is specified in DATAFORMAT. Therefore, minimum usage can be achieved, with a viable input *tfdata*, by:

.. code-block:: Python

    from brooksrfigan import training
    generator_model, discriminator_model = training.train(tfdata)

