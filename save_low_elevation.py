# %% [markdown]
# # Reading in Low fidelity (elevation and Temperature) and high fidelity data (surface temperature, elevation)

# %%
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
print(hi_elev.shape)
lo_t2_arr.shape
test = xr.open_dataset('data/hifid_hgt.nc')
hi_t2_arr.shape[0]*0.5

# %%
def create_netcdf(filename, time, latitude, longitude, data):
    
    
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
    data_var[:,:,:] = data

    # Add variable attributes
    #time_var.units = "hours since 0001-01-01 00:00:00"
    lat_var.units = "degrees"
    lon_var.units = "degrees"

    # Close the NetCDF file
    nc_file.close()


# %% [markdown]
# # Define Neural Network trained with the high fidelity temperature and elevation data, and use the low fidelity temperature data to predict the low fidelity elevation

# %%
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

    # Padding low-resolution temperature data with zeros
    padded_temp_data = np.zeros(temp_data.shape)
    padded_temp_data[:temp_data_lowres.shape[0],:temp_data_lowres.shape[1]] = temp_data_lowres
    
    # Predict elevation using the padded low-resolution temperature data
    elev_pred = model.predict(padded_temp_data)

    # Slice the elevation prediction values to the correct shape
    elev_pred_lowres = elev_pred[:temp_data_lowres.shape[0],:temp_data_lowres.shape[1]]
    
    return elev_pred_lowres
    
temp = np.array([]) # this is an array that will store

new_hi_t2_arr = hi_t2_arr.copy()
#hi_t2_arr = hi_t2_arr[:3,:,:]  # this is to test over a smaller set
# these for loops are to generate temp array that will store the low fidelity elevation elevation profile



for i in range(hi_t2_arr.shape[0]):
#for i in range(1):
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

time = time_dim
latitude = lo_lats
longitude = lo_lons
data = lo_elev

# print time latitude longitude and data shape
print('time shape: ', time.shape)

print('latitude shape: ', latitude.shape)

print('longitude shape: ', longitude.shape)

print('data shape: ', data.shape)

create_netcdf("low_fidelity_elevation_3.nc", time, latitude, longitude, data)



