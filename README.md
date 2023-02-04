# Application of Neural Networks for Climate Downscaling

## Introduction

Climate data downscaling is a process of estimating the impact of climate change at a local scale by combining large-scale climate models with high-resolution observations. The data folder for this application contains the following data:

- Elevation data at high resolution
- High- and low-fidelity temperature data
- Low-resolution wind data

The purpose of this project is to use python scripts to estimate high-fidelity wind data (u10 and v10) using neural networks. 

## Methods 

> save_low_elevation.py (make this a link )
>- trains a network with the highe resolution Temperature and elevation data. 
>- Using the Deep learning model, we use the low resolution temperature data to estimate the low resolution elevation data

> hires_u10_v10.ipynb (make this a link)
>- train a deep Neural network with low resolution wind and elevation data
>- feed the model with high resolution elevation data to estimate high resolution U10 and V10 (data saved in the output folder)

> error_nmbf_estimate.ipynb
>- estimating normalised mean bias factor between low resolution elevation data estimated at every time step. 

>low_fidelity_elevation_3.nc : is the netcdf file containing the low resolution elevation data which was estimted using save_low_elevation.py


## Data Source

The data used in this project is taken from [https://github.com/scotthosking/mf_modelling](https://github.com/scotthosking/mf_modelling). 

## Conclusion

This project highlights the potential of neural networks in downscaling climate data, which is an important step in understanding the impact of climate change at a local scale. 
