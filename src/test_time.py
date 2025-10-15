import xarray as xr

ten_day = "./restart3/DINO_1m_grid_T_2D.nc"
ds = xr.open_dataset(ten_day)
ds_small = ds.isel(time_counter=slice(0, 30))

# get time coordinate
time = ds_small.coords["time_counter"]
print(time)

# now resample to monthly

print("now resample")
monthly = ds_small.resample(time_counter="3ME").mean()
monthly.load()
# get time coordinate
time = monthly.coords["time_counter"]
print(time)

# Now load the 1m data and see if it's different:
one_month = "./restart3/DINO_3m_grid_U_3D.nc"
ds_month = xr.open_dataset(one_month)
ds_small = ds_month["uoce"].isel(time_counter=slice(0, 10))

# get time coordinate
time = ds_small.coords["time_counter"]
print(time)
