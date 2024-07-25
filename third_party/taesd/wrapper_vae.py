"""
This is a wrapper for the TAESD VAE
"""

import tensorflow as tf
import numpy as np
import torch

class WrapperVAE(tf.keras.Model):
    def __init__(self, taesd_encoder, **kwargs):
        super(WrapperVAE, self).__init__(**kwargs)
        self.taesd_encoder = taesd_encoder

    # def call(self, inputs, training=False):
    #     inputs = tf.convert_to_tensor(inputs)
    #     inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])  # NHWC to NCHW for PyTorch
    #     inputs = tf.identity(inputs)  # Ensure it's a float32 tensor
    #     encoded = tf.py_function(func=self.encode, inp=[inputs], Tout=tf.float32)
    #     return encoded
    
    def call(self, inputs, training=False):
        inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])  # NHWC to NCHW for PyTorch
        
        # Use tf.py_function to wrap the PyTorch model call
        encoded = tf.py_function(func=self.encode, inp=[inputs], Tout=tf.float32)
        
        # Explicitly set the shape of output tensor
        encoded.set_shape([None, 4, None, None])  # Assuming latent_channels=4, modify as necessary
        
        # Transpose back from NCHW to NHWC
        return tf.transpose(encoded, perm=[0, 2, 3, 1])

    def encode(self, inputs):
        with torch.no_grad():
            inputs = torch.tensor(inputs.numpy())
            encoded = self.taesd_encoder.encoder(inputs)
        return encoded.detach().numpy()