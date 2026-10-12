"""las_to_recarray.py

High-performance LAS/LAZ → NumPy structured array utilities.

Features
--------
- Full load into NumPy recarray
- Chunked streaming for large files
- Memory-mapped output for huge datasets
- Automatic scaled coordinates (x, y, z)
- Field selection for memory efficiency

Requirements
------------
pip install laspy numpy
pip install lazrs   # for LAZ support
"""

from typing import Iterator, Optional, Tuple, List
import numpy as np
import laspy
import os
import logging


# -------------------------------------------------------
# DTYPE CONSTRUCTION
# -------------------------------------------------------
def _build_dtype(
    point_format, fields: Optional[List[str]] = None, include_scaled: bool = True
) -> Tuple[np.dtype, List[str]]:
    """Build NumPy dtype with optional field selection and scaled coords."""

    available = {dim.name: dim for dim in point_format.dimensions}

    if fields is None:
        selected = list(available.keys())
        logging.debug((selected))
    else:
        selected = list(fields)
        missing = [f for f in selected if f not in available]
        if missing:
            pass
            # raise ValueError(f"Fields not in LAS file: {missing}")

    dtype_fields = []

    for name in selected:
        if name in available:
            dim = available[name]
        else:
            np_dtype = np.float64
        try:
            np_dtype = dim.dtype
        except AttributeError:
            np_dtype = np.float64
        dtype_fields.append((name, np_dtype))

    if include_scaled:
        dtype_fields.extend(
            [
                ("xyz", (np.float64, 3)),
            ]
        )

    return np.dtype(dtype_fields), selected


# -------------------------------------------------------
# CORE CONVERSION
# -------------------------------------------------------
def _fill_array(arr: np.ndarray, points, header, selected_fields, include_scaled: bool):
    """Fill structured array from LAS points"""

    for name in selected_fields:
        try:
            arr[name] = points[name]
        except:
            pass
    if include_scaled:
        arr["xyz"][:, 0] = points.x
        arr["xyz"][:, 1] = points.y
        arr["xyz"][:, 2] = points.z


def _chunk_to_recarray(
    points, dtype, header, selected_fields, include_scaled
) -> np.recarray:
    n = len(points)
    arr = np.empty(n, dtype=dtype)

    _fill_array(arr, points, header, selected_fields, include_scaled)

    return arr.view(np.recarray)


# -------------------------------------------------------
# 1. FULL LOAD
# -------------------------------------------------------
def las_to_recarray(
    file_path: str, fields: Optional[List[str]] = None, include_scaled: bool = True
) -> np.recarray:
    """Load entire LAS/LAZ file into NumPy recarray."""

    las = laspy.read(file_path)

    dtype, selected_fields = _build_dtype(las.point_format, fields, include_scaled)

    arr = np.empty(len(las.points), dtype=dtype)

    _fill_array(arr, las.points, las.header, selected_fields, include_scaled)

    return arr.view(np.recarray)


# -------------------------------------------------------
# 2. CHUNKED READING
# -------------------------------------------------------
def iter_las_chunks(
    file_path: str,
    chunk_size: int = 1_000_000,
    fields: Optional[List[str]] = None,
    include_scaled: bool = True,
) -> Iterator[np.recarray]:
    """Stream LAS/LAZ file in chunks as recarrays."""

    with laspy.open(file_path) as f:
        dtype, selected_fields = _build_dtype(
            f.header.point_format, fields, include_scaled
        )

        for points in f.chunk_iterator(chunk_size):
            yield _chunk_to_recarray(
                points, dtype, f.header, selected_fields, include_scaled
            )


# -------------------------------------------------------
# 3. MEMORY-MAPPED OUTPUT
# -------------------------------------------------------
def las_to_memmap(
    file_path: str,
    output_path: str,
    fields: Optional[List[str]] = None,
    include_scaled: bool = True,
    chunk_size: int = 1_000_000,
    overwrite: bool = True,
) -> np.memmap:
    """Convert LAS/LAZ into memory-mapped NumPy structured array."""

    with laspy.open(file_path) as f:
        n_points = f.header.point_count

        dtype, selected_fields = _build_dtype(
            f.header.point_format, fields, include_scaled
        )

        if os.path.exists(output_path):
            if overwrite:
                os.remove(output_path)
            else:
                raise FileExistsError(output_path)

        mm = np.memmap(output_path, dtype=dtype, mode="w+", shape=(n_points,))

        offset = 0

        for chunk in f.chunk_iterator(chunk_size):
            rec = _chunk_to_recarray(
                chunk, dtype, f.header, selected_fields, include_scaled
            )

            size = len(rec)
            mm[offset : offset + size] = rec
            offset += size

        mm.flush()
        return mm


# -------------------------------------------------------
# 4. PROCESSING HELPER
# -------------------------------------------------------
def process_las_in_chunks(
    file_path: str,
    func,
    chunk_size: int = 1_000_000,
    fields: Optional[List[str]] = None,
    include_scaled: bool = True,
):
    """Apply a function to chunks (map-reduce style)."""

    results = []

    for chunk in iter_las_chunks(file_path, chunk_size, fields, include_scaled):
        results.append(func(chunk))

    return results


# -------------------------------------------------------
# 5. SCHEMA INSPECTION
# -------------------------------------------------------
def print_schema(arr: np.ndarray):
    """Print structured dtype schema."""
    for name in arr.dtype.names:
        logging.debug((f"{name}: {arr.dtype[name]}"))


# -------------------------------------------------------
# EXAMPLE USAGE
# -------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        logging.debug(("Usage: python las_to_recarray.py <file.las>"))
        exit(1)

    path = sys.argv[1]

    logging.debug(("\n== Full load =="))
    rec = las_to_recarray(path, fields=["X", "Y", "Z"], include_scaled=True)
    logging.debug(("Points:", len(rec)))
    print_schema(rec)

    logging.debug(("\n== Chunked example =="))
    for i, chunk in enumerate(iter_las_chunks(path, chunk_size=500_000)):
        logging.debug((f"Chunk {i}: {len(chunk)} points"))
        if i == 2:
            break

    logging.debug(("\n== Memmap example =="))
    mm = las_to_memmap(
        path, "points.dat", fields=["X", "Y", "Z", "intensity"], include_scaled=True
    )
    logging.debug(("Memmap shape:", mm.shape))
