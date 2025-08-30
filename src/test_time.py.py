import xarray as xr

ten_day = "./restart3/DINO_10d_grid_inst_T_3D.nc"
ds = xr.open_dataset(ten_day)
ds_small = ds.isel(time_counter=slice(0, 10))

# get time coordinate
time = ds_small.coords["time_counter"]
print(time)

# now resample to monthly
monthly = ds_small.resample(time_counter="1ME").mean()
# get time coordinate
time = monthly.coords["time_counter"]
print(time)
