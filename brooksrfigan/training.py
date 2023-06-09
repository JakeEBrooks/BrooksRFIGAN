import os
import time
import datetime
import logging
log = logging.getLogger(__name__)

import numpy as np
import tensorflow as tf

from brooksrfigan.generator import Unet_default
from brooksrfigan.discriminator import ConvNet_default

bce_loss = tf.keras.losses.BinaryCrossentropy()
mae_loss = tf.keras.losses.MeanAbsoluteError()

train_img_metric = tf.keras.metrics.MeanAbsoluteError()
train_gen_metric = tf.keras.metrics.BinaryCrossentropy()
train_disc_metric = tf.keras.metrics.BinaryCrossentropy()

val_img_metric = tf.keras.metrics.MeanAbsoluteError()
val_gen_metric = tf.keras.metrics.BinaryCrossentropy()
val_disc_metric = tf.keras.metrics.BinaryCrossentropy()

generator_optimizer = tf.keras.optimizers.Adam()
discriminator_optimizer = tf.keras.optimizers.Adam()

def s_to_hms(total_seconds):
    """Convenience function to get N seconds in HMS format"""
    hours = total_seconds // 3600
    minutes = (total_seconds - hours*3600) // 60
    seconds = (total_seconds - hours*3600 - minutes*60)
    return '{0}h {1}m {2}s'.format(int(hours), int(minutes), int(seconds))

def generator_loss(disc_fake_output, generated_masks, real_masks, lam):
    gan_loss = bce_loss(tf.ones_like(disc_fake_output), disc_fake_output)
    image_loss = mae_loss(generated_masks, real_masks)
    return gan_loss + lam*image_loss

def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = bce_loss(tf.ones_like(disc_real_output), disc_real_output)
    fake_loss = bce_loss(tf.zeros_like(disc_fake_output), disc_fake_output)
    return real_loss + fake_loss

@tf.function
def train_step(im_batch, mask_batch, generator, discriminator, gen_loss_lambda):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_masks = generator(im_batch, training=True)
        disc_real_out = discriminator(mask_batch, training=True)
        disc_fake_out = discriminator(generated_masks, training=True)

        gen_loss = generator_loss(disc_fake_out, generated_masks, mask_batch, gen_loss_lambda)
        disc_loss = discriminator_loss(disc_real_out, disc_fake_out)

    generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))

    train_img_metric.update_state(mask_batch, generated_masks) # How good the masks look
    train_gen_metric.update_state(tf.ones_like(disc_fake_out), disc_fake_out) # How good the generator is at tricking the discriminator
    train_disc_metric.update_state(tf.zeros_like(disc_fake_out), disc_fake_out) # How good the discriminator is at spotting fakes

@tf.function
def validate_step(im_batch, mask_batch, generator, discriminator):
    generated_masks = generator(im_batch, training=False)
    disc_fake_out = discriminator(generated_masks, training=False)

    val_img_metric.update_state(mask_batch, generated_masks) # How good the images look
    val_gen_metric.update_state(tf.ones_like(disc_fake_out), disc_fake_out) # How good the generator is at tricking the discriminator
    val_disc_metric.update_state(tf.zeros_like(disc_fake_out), disc_fake_out) # How good the discriminator is at spotting fakes


def train(dataset, epochs=10, batch_size=32, validation_multi=1, gen_loss_lambda=100, generator_model=None, discriminator_model=None, tblogdir='./tensorboard_log/',enable_tensorboard=False):
    """
    The primary function of this package. Interfaces with Tensorflow to produce a trained generator model and discriminator model. The trained generator model can be used generate accurate flag masks that
    retain the characteristics of those found in the training set. By default, the function will train a generator and a discriminator using the basic models found in brooksrfigan.generator and brooksrfigan.dsicriminator,
    though an option exists to specify a different Keras model. Details of the training process can be logged to tensorboard through the *enable_tensorboard* parameter, and the location of the stored information is 
    controlled by passing a filepath to *tblogdir*. Also outputs useful information to the python logger during training.

    .. role:: python(code)
        :language: python

    :param dataset: A two component `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_ instance containing the image and ground truth mask cutouts. Normally created by something like:
        :python:`dataset = tf.data.Dataset.from_tensor_slices((images,masks))` where *images* and *masks* are numpy arrays created by preprocessing.make_cutouts, for example.
    :param epochs: An integer specifying the number of loops to perform over the full training set. A validation run is performed at the end of each epoch.
    :param batch_size: How many images to process in each training step. The choice of batch_size is pretty much down to how much memory you have on your system. Higher RAM capacity systems can process more batches at once.
    :param validation_multi: How many multiples of *batch_size* to set aside as a validation set. For example, with a batch size of 16, setting *validation_multi = 4* would keep aside 64 images for validation at
        the end of each epoch
    :param gen_loss_lambda: An integer specifying the weight associated with the mean absolute error in the generator loss. Leave this as default unless you know what you're doing!
    :param generator_model: Accepts a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`_ instance for a custom generator model. Leaving unspecified will use the model defined in brooksrfigan.generator.
    :param discriminator_model: Same as *generator_model*, but for a discriminator. Leaving unspecified will use the model defined in brooksrfigan.discriminator.
    :param tblogdir: A string indicating the filepath to store tensorboard information. Defaults to the current working directory.
    :param enable_tensorboard: A toggle option for tensorboard logging.
    :returns: trained generator model, trained discriminator model
    """
    if generator_model != None and isinstance(generator_model, tf.keras.Model):
        generator = generator_model
    else:
        generator = brooksrfigan.generator.Unet_default((128,1024,1))
    if discriminator_model != None and isinstance(discriminator_model, tf.keras.Model):
        discriminator = discriminator_model
    else:
        discriminator = brooksrfigan.discriminator.ConvNet_default((128,1024,1))

    if enable_tensorboard:
        tbsamples = dataset.take(3).batch(3)
        for x, y in tbsamples: #this is so dumb, do better tensorflow
            tbimages = x
            tbmasks = y
        tstart = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_writer = tf.summary.create_file_writer(tblogdir+tstart+'/Training_Metrics')
        val_writer = tf.summary.create_file_writer(tblogdir+tstart+'/Validation_Metrics')
        img_writer = tf.summary.create_file_writer(tblogdir+tstart+'/Generator_Guesses')
        with img_writer.as_default():
            tf.summary.image("Input Images", tbimages, max_outputs=3, step=0)
            tf.summary.image("Input Masks", tbmasks, max_outputs=3, step=0)

    log.info('-- Splitting data into training and validation sets...')
    val_ds = dataset.take(int(batch_size*validation_multi))
    train_ds = dataset.skip(int(batch_size*validation_multi))
    val_ds = val_ds.shuffle(10, reshuffle_each_iteration=False).batch(batch_size)
    train_ds = train_ds.shuffle(10, reshuffle_each_iteration=False).batch(batch_size)

    train_starttime = time.time()
    for epoch in range(epochs):
        epoch_startime = time.time()
        log.info('')
        log.info('---- Starting training epoch {}/{}'.format(epoch+1, epochs))

        for step, (im_batch, mask_batch) in enumerate(train_ds):
            train_step(im_batch, mask_batch, generator, discriminator, gen_loss_lambda)
            log.info('Iterations - {}   Generated image accuracy - {:.4f}   Generator performance - {:.4f}   Discriminator performance - {:.4f}'.format(step, train_img_metric.result(), train_gen_metric.result(), train_disc_metric.result()))
        if enable_tensorboard:
            with train_writer.as_default(step=epoch):
                tf.summary.scalar('Image MAE',train_img_metric.result())
                tf.summary.scalar('Generator Performace',train_gen_metric.result())
                tf.summary.scalar('Discriminator Performance',train_disc_metric.result())
        train_img_metric.reset_state()
        train_gen_metric.reset_state()
        train_disc_metric.reset_state()

        log.info('-- Finished training iterations, validating...')
        for step, (im_batch, mask_batch) in enumerate(val_ds):
            validate_step(im_batch, mask_batch, generator, discriminator)
        log.info('Validation metrics:  Generated image accuracy - {:.4f}   Generator performance - {:.4f}   Discriminator performance - {:.4f}'.format(val_img_metric.result(), val_gen_metric.result(), val_disc_metric.result()))
        if enable_tensorboard:
            with val_writer.as_default(step=epoch):
                tf.summary.scalar('Image MAE',val_img_metric.result())
                tf.summary.scalar('Generator Performace',val_gen_metric.result())
                tf.summary.scalar('Discriminator Performance',val_disc_metric.result())
        val_img_metric.reset_state()
        val_gen_metric.reset_state()
        val_disc_metric.reset_state()

        if enable_tensorboard:
            with img_writer.as_default():
                tf.summary.image("Generator guesses", generator(tbimages, training=False), max_outputs=3, step=epoch)

        log.info('-- Time taken for this epoch: {}'.format(s_to_hms(time.time() - epoch_startime)))
        if epoch < (epochs - 1):
            log.info('-- ETA: {}'.format(s_to_hms((time.time() - epoch_startime)*((epochs - 1) - epoch))))
    
    log.info('')
    log.info('Finished training in {}'.format(s_to_hms(time.time() - train_starttime)))

    return generator, discriminator