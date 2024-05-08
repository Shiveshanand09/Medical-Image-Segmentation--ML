import numpy
import keras
import keras.backend as K

#Dice and IoU loss fucntion

def DiceLoss(targets, inputs, smooth=1e-6):

  #inputs = K.flatten(inputs)
  #targets = K.flatten(targets)
  targets = K.batch_flatten(targets)
  inputs = K.batch_flatten(inputs)

  intersection = K.sum(targets * inputs, axis=-1)
  #intersection = K.sum(targets * inputs)
  dice = (2 * intersection + smooth) / (K.sum(targets, axis=-1) + K.sum(inputs, axis=-1) + smooth)
  return 1 - dice

def IoULoss(targets, inputs, smooth=1e-6):
  targets = K.batch_flatten(targets)
  inputs = K.batch_flatten(inputs)

  intersection = K.sum(targets * inputs, axis=-1)
  union = K.sum(targets) + K.sum(inputs) - intersection


  IoU = (intersection + smooth) / (union + smooth)
  return 1 - IoU
