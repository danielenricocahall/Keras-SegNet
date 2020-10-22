# Keras-SegNet #
Keras implementation of the image segmentation architecture, SegNet (https://arxiv.org/abs/1511.00561).

![alt text](https://github.com/danielenricocahall/Keras-SegNet/blob/master/Figures/segnet.png)

# Setup

`pipenv install .` should configure a python environment and install all necessary dependencies in the environment. 

# Structure

Custom layers are defined in the `custom_layers` package in the `layers` module. A function for creating an example SegNet model is in the `segnet ` package in the `create_segnet` module.

# Testing

Tests for both custom layers are defined in `test/test_layers.py`. Execute `pytest test` to run.





