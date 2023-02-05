# Application of Neural Networks for Climate Downscaling

## Introduction

Climate data downscaling is a process of estimating the impact of climate change at a local scale by combining large-scale climate models with high-resolution observations. The [data](data) folder for this application contains the following data:

- Elevation data at high resolution
- High- and low-fidelity temperature data
- Low-resolution wind data

The purpose of this project is to use python scripts to estimate high-fidelity wind data (u10 and v10) using neural networks. 

## Methods 

[save_low_elevation.py](save_low_elevation.py)
- Trains a network with the high resolution temperature and elevation data.
- Using the deep learning model, we use the low resolution temperature data to estimate the low resolution elevation data.
- [error_nmbf_estimate.ipynb](error_nmbf_estimate.ipynb) Estimates the normalized mean bias factor between low resolution elevation data estimated at every time step. This script was just to for testing. 

[hires_u10_v10.ipynb](hires_u10_v10.ipynb)
- Trains a deep neural network with low resolution wind and elevation data.
- Feeds the model with high resolution elevation data to estimate high resolution U10 and V10 (data saved in the [output](output) folder).



[low_fidelity_elevation_3.nc](low_fidelity_elevation_3.nc): is the netCDF file containing the low resolution elevation data which was estimated using [save_low_elevation.py](save_low_elevation.py).


## Data Source

The data used in this project is taken from [https://github.com/scotthosking/mf_modelling](https://github.com/scotthosking/mf_modelling). 

## Conclusion

This project highlights the potential of neural networks in downscaling climate data, which is an important step in understanding the impact of climate change at a local scale. 
