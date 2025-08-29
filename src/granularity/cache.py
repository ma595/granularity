import os
import hashlib
import json


def show_cache_stats(cache_dir="./resampled_cache"):
    """Show cache statistics"""
    if not os.path.exists(cache_dir):
        print("No disk cache found")
        return

    nc_files = [f for f in os.listdir(cache_dir) if f.endswith(".nc")]
    if not nc_files:
        print("Cache directory exists but is empty")
        return

    total_size = 0
    cached_items = []

    for nc_file in nc_files:
        file_path = os.path.join(cache_dir, nc_file)
        size = os.path.getsize(file_path)
        total_size += size

        # Parse filename: "temperature_10d_to_1m.nc"
        name_parts = nc_file[:-3].split("_to_")
        if len(name_parts) == 2:
            source_parts = name_parts[0].split("_")
            if len(source_parts) >= 2:
                var = "_".join(source_parts[:-1])
                source_gran = source_parts[-1]
                target_gran = name_parts[1]
                cached_items.append((var, source_gran, target_gran, size))

    print(f"\n=== DISK CACHE STATISTICS ===")
    print(f"Cache directory: {cache_dir}")
    print(f"Total files: {len(nc_files)}")
    print(f"Total size: {total_size / 1024**3:.2f} GB")

    if cached_items:
        print("\nCached resampled data:")
        for var, source_gran, target_gran, size in sorted(cached_items):
            print(f"  {var}: {source_gran}→{target_gran} ({size / 1024**2:.1f} MB)")
    else:
        print("No valid cached items found")


def get_cache_filename(var, source_gran, target_gran, cache_dir="./resampled_cache"):
    """Generate consistent cache filename"""
    cache_key = f"{var}_{source_gran}_to_{target_gran}"
    return os.path.join(cache_dir, f"{cache_key}.nc")


def get_file_hash(filepath):
    """Get hash of source file to detect changes"""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read(1024 * 1024)).hexdigest()  # Hash first 1MB for speed


def write_metadata(cache_file, meta: dict):
    meta_file = cache_file.replace(".nc", "_metadata.json")
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)
