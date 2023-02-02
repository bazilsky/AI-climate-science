#!/usr/bin/env python
# coding: utf-8

# In[12]:


import xarray as xr
import iris
import numpy as np 
import pylab as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
import netCDF4 as nc


# In[13]:



### Regional gridded output from a Regional Climate Model (RCM)
hi_t2   = xr.open_dataset('data/hifid_t2m_monthly.nc')
hi_lats = hi_t2['latitude'].values
hi_lons = hi_t2['longitude'].values 

### Global gridded reanalysis (data assimilated) used to update/force the 
### lateral boundaries of the RCM
lo_t2  = xr.open_dataset('data/lofid_t2m_monthly.nc')
lo_u10 = xr.open_dataset('data/lofid_u10_monthly.nc')
lo_v10 = xr.open_dataset('data/lofid_v10_monthly.nc')

### work with numpy arrays
hi_t2_arr  = hi_t2['t2m'].values
lo_t2_arr  = lo_t2['t2m'].values 
lo_u10_arr = lo_u10['u10'].values
lo_v10_arr = lo_v10['v10'].values
time_dim   = lo_t2['time'].values 

##low fidelity lat and lon 
lo_lats = lo_t2['latitude'].values
lo_lons = lo_t2['longitude'].values

# high fidelity elevation data 
hi_elev = xr.open_dataset('data/hifid_hgt.nc')['hgt'].values


# In[14]:


def train_temp_model(temp_data, elevation_data, temp_data_lowres):
    # Convert data to float32
    temp_data = temp_data.astype(np.float32)
    elevation_data = elevation_data.astype(np.float32)

    # Define the model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(temp_data.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(elevation_data.shape[1]))
    

    # Compile the model
    #model.compile(optimizer='adam', loss='mean_squared_error')
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(temp_data, elevation_data, epochs=100, batch_size=32, verbose = False) # verbose = 0 showing no training progress
    
    old_shape = temp_data_lowres.shape
    new_shape = temp_data.shape

    padded_array = np.zeros(new_shape)
    padded_array[:old_shape[0],:old_shape[1]] = temp_data_lowres
    
    elev_pred = model.predict(padded_array)
    
    elev_pred_slice = elev_pred[:old_shape[0],:old_shape[1]]
    #model.evaluate(temp_data,elevation_data)
    #print ('accuracy = ', accuracy)
    return elev_pred_slice

# In[15]:


temp = np.array([]) # this is an array that will store

# these for loops are to generate temp array that will store the low fidelity elevation elevation profile

for i in range(hi_t2_arr.shape[0]):
#for i in range(2):
    pred = train_temp_model(hi_t2_arr[i,:,:], hi_elev, lo_t2_arr[i,:,:])
    pred = pred[np.newaxis,:,:] # add a new axis
    if i ==0:
        temp = pred.copy()
    else:
        temp = np.concatenate([temp,pred], axis = 0)
    print(f'value os i is {i} and max val = {hi_t2_arr.shape[0]}')
lo_elev = temp
print(hi_t2_arr.shape)
print(lo_t2_arr.shape)
print(pred.shape)

#save_netcdf(temp)


# In[16]:


def create_netcdf(filename, time, latitude, longitude, data):
    
    #filename = 'low_fidelity_elevation.nc'
    filepath = os.path.join(os.getcwd(),filename)
    flag = os.path.exists(filepath)
    
    if flag == True:
        os.remove(filename)
    
    # Open a new NetCDF file
    nc_file = nc.Dataset(filename, "w", format="NETCDF4")

    # Create dimensions
    nc_file.createDimension("time", len(time))
    nc_file.createDimension("latitude", len(latitude))
    nc_file.createDimension("longitude", len(longitude))

    # Create variables
    time_var = nc_file.createVariable("time", "f8", ("time",))
    lat_var = nc_file.createVariable("latitude", "f4", ("latitude",))
    lon_var = nc_file.createVariable("longitude", "f4", ("longitude",))
    data_var = nc_file.createVariable("lo_elev", "f4", ("time", "latitude", "longitude"))

    # Write data to variables
    time_var[:] = time
    lat_var[:] = latitude
    lon_var[:] = longitude
    data_var[:] = data

    # Add variable attributes
    #time_var.units = "hours since 0001-01-01 00:00:00"
    lat_var.units = "degrees"
    lon_var.units = "degrees"

    # Close the NetCDF file
    nc_file.close()



time = time_dim
latitude = lo_lats
longitude = lo_lons
data = pred
create_netcdf("low_fidelity_elevation.nc", time, latitude, longitude, data)





