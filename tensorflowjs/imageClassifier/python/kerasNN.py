# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 12:06:21 2019

@author: chrisxt
"""

import keras
import tensorflowjs as tfjs

vgg16= keras.applications.vgg16.VGG16()

tfjs.converters.save_keras_model(vgg16,'/home/chrisxt/Documents/TFJS_tutorial_v2/imageClassifier/client/vgg/')