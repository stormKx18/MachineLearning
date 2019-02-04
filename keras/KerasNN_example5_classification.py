# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 19:10:51 2019
Source: [Jojo_John_Moolayil]_Learn_Keras_for_Deep_Neural_N.pdf
@author: chrisxt
Objective: DNN model used for classification
"""
#----------------------------------------------------------------------------------------------------------------
#Load data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import History
#----------------------------------------------------------------------------------------------------------------
