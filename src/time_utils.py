import numpy as np
import cftime

import xarray as xr


def _to_days(arr, calendar=None):
    """Return float days for a 1D array of times (numpy or cftime)."""
    v = xr.DataArray(arr).values
    if v.size == 0:
        return np.array([], dtype=float)

    # NumPy datetime64 case
    if np.issubdtype(v.dtype, np.datetime64):
        # cast to whole days since 1970-01-01 and to float
        return v.astype("datetime64[D]").astype("int64").astype(float)

    # cftime case (e.g., 360_day, noleap, gregorian, ... incl. year 0001)
    cal = calendar
    if cal is None:
        # try to infer calendar from the array/index
        cal = (
            getattr(getattr(arr, "calendar", None), "name", None)
            or getattr(arr, "calendar", None)
            or "standard"
        )
    return cftime.date2num(
        v.tolist(), units="days since 0001-01-01", calendar=cal
    ).astype(float)


def inspect_time_labels(ds, time_dim=None, name=None):
    td = time_dim or ("time" if "time" in ds.dims else "time_counter")
    t = ds[td]
    bname = t.attrs.get("bounds", None)

    if bname and bname in ds:
        b = ds[bname]  # shape (time, 2); last dim are bounds
        cal = getattr(getattr(t.indexes, td, None), "calendar", None)

        lab_days = _to_days(t, calendar=cal)
        beg_days = _to_days(b.isel({b.dims[-1]: 0}), calendar=cal)
        end_days = _to_days(b.isel({b.dims[-1]: 1}), calendar=cal)
        mid_days = 0.5 * (beg_days + end_days)

        d_lab_mid = np.abs(lab_days - mid_days).mean()
        d_lab_beg = np.abs(lab_days - beg_days).mean()
        d_lab_end = np.abs(lab_days - end_days).mean()

        if d_lab_mid <= min(d_lab_beg, d_lab_end):
            pos = "mid-interval"
        elif d_lab_beg <= d_lab_end:
            pos = "start"
        else:
            pos = "end"

        print(f"[{name or 'dataset'}] {td}: label={pos}, bounds={bname}, n={t.size}")
    else:
        print(
            f"[{name or 'dataset'}] {td}: no bounds; cannot prove start/mid/end. n={t.size}"
        )
