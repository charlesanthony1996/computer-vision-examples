import tensorflow as tf
import torch
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import datetime
import os
import re
import io
import random
# from google.colab import drive, files
from PIL import Image
import albumentations as A
import tensorflow_datasets as tfds
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, InputLayer, BatchNormalization, Input, Dropout, RandomFlip, RandomRotation, Resizing, Rescaling
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, FalsePositives, FalseNegatives, TruePositives, TrueNegatives, Precision, Recall, AUC, binary_accuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, CSVLogger, EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2, L1
from tensorboard.plugins.hparams import api as hp
# from google.colab import drive, files


# wandb installation
# weights and biases module for python


# import wandb
# from wandb.keras import WandbCallback

# wandb.init(project="Malaria-Detection", entity="neuralearn")
# wandb.init(project="Malaria-Detection", entity="neuralearn", sync_tensorboard = True)


# wandb.config = {
#     "LEARNING_RATE": 0.001,
#     "N_EPOCHS": 5,
#     "BATCH_SIZE": 128,
#     "DROPOUT_RATE": 0.0,
#     "IM_SIZE": 224,
#     "REGULARIZATION_RATE": 0.0,
#     "N_FILTERS": 6,
#     "KERNEL_SIZE": 3,
#     "N_STRIDES": 1,
#     "POOL_SIZE": 2,
#     "N_DENSE_1": 100,
#     "N_DENSE_2": 10,
# }

# CONFIGURATION = wandb.config

# data preparation
# data loading


# download the dataset

