"""Input/output utilities for LiDAR datasets, including LAS/LAZ parsing,
trajectory file reading, RGB/range/intensity image saving with EXIF/GPS
metadata, and PCD point cloud writing.

This module deals exclusively with data ingestion and export. It contains
no geometric logic or projection code.
"""

import os
import tempfile
import logging
import numpy as np
import laspy
import cv2
from PIL import Image
import matplotlib as mpl

from PIL import ExifTags

from PIL.ExifTags import GPS

from pypcd4 import PointCloud
from .pyprojCoordinateTranformer import ETRSTM35FINxy_to_WGS84lalo
from .x200go_nmea_parser import parse_nmea_file, parse_pose_yaml

from .utils import points_dtype, MemmapMetadata

from .las_helper import las_to_memmap, las_to_recarray


# -------------------------------------------------------------------------
# LAS/LAZ reading
# -------------------------------------------------------------------------
def load_pointcloud(pointcloud_file, fields=["intensity", "classification"]):
    point_cloud, _, _ = read_laz_file(pointcloud_file, memmap=False, fields=fields)

    # Keep a clean copy for each iteration
    #     original_pointcloud = np.zeros(
    #         (point_cloud.shape,5 )
    #    )  # add ID field

    lenght = 3 + len(fields)
    field_mapping = {}

    tmp_file, orginal_pc_mmap_file = tempfile.mkstemp(dir=".")
    orginal_pc_mmap = np.memmap(
        orginal_pc_mmap_file,
        dtype=points_dtype,
        mode="w+",
        shape=(point_cloud.shape[0], lenght),
    )
    orginal_pc_mmap_meta = MemmapMetadata(
        filepath=orginal_pc_mmap_file,
        dtype=orginal_pc_mmap.dtype,
        shape=orginal_pc_mmap.shape,
    )
    orginal_pc_mmap[:, 0:3] = point_cloud["xyz"]

    i = 3
    try:
        intensity = 65534 * (point_cloud["intensity"] / np.max(point_cloud["intensity"]))
        orginal_pc_mmap[:, i] = intensity  # point_cloud["intensity"]
        field_mapping["intensity"] = i
        i += 1
    except ValueError:
        logging.debug(("intensity error"))
        pass
    try:
        orginal_pc_mmap[:, i] = point_cloud["classification"]
        field_mapping["classification"] = i
        i += 1
    except ValueError:
        logging.debug(("classification error"))
        pass
    try:
        orginal_pc_mmap[:, i] = point_cloud["id"]
        field_mapping["id"] = i
        i += 1
    except ValueError:
        logging.debug(("id error"))

        pass

    try:
        orginal_pc_mmap[:, i] = point_cloud["red"]
        orginal_pc_mmap[:, i + 1] = point_cloud["green"]
        orginal_pc_mmap[:, i + 2] = point_cloud["blue"]
        i += 3
    except ValueError:
        logging.debug(("COLOR error"))

    orginal_pc_mmap.field_mapping = field_mapping
    return orginal_pc_mmap_meta


def read_laz_file(file_path, memmap=True, chunk_size=10_000_000, fields=None):
    """Read a .laz or .las file and return a memory‑mapped Nx8 array holding:

        x, y, z, intensity, classification, red, green, blue

    If color channels are missing, falls back to using the normalized
    intensity channel as a pseudo‑RGB signal.

    Parameters
    ----------
    file_path : str
        Path to the LAZ/LAS file.

    Returns
    -------
    numpy.memmap
        Mapped array of shape (N, 8).
    """
    if memmap:
        tmp_file, filename = tempfile.mkstemp(dir=".")
        pc = las_to_memmap(file_path, filename, chunk_size, fields=fields)
        return pc, tmp_file, filename

    pc = las_to_recarray(file_path, fields=fields)
    return pc, None, None

    las = laspy.read(file_path)

    # Normalize 16-bit intensity to 0..65535 range
    # 2^16 = 65536
    intensity = 65534 * (las.intensity / np.max(las.intensity))

    # try:
    #     red = las.red
    #     green = las.green
    #     blue = las.blue
    # except Exception:
    #     red = green = blue = intensity
    if not memmap:
        pc = np.zeros(shape=(len(las.x), 4), dtype=points_dtype)
        pc[:, 0] = las.x
        pc[:, 1] = las.y
        pc[:, 2] = las.z
        pc[:, 3] = intensity
        # pc[:, 4] = las.classification
        # pc[:, 5] = red
        # pc[:, 6] = green
        # pc[:, 7] = blue
        return pc, None, None

    if memmap:
        tmp_file, filename = tempfile.mkstemp(dir=".")
        pc_mmap = np.memmap(
            filename,
            dtype=points_dtype,
            mode="w+",
            shape=(len(las.x), 4),
        )
        pc_mmap[:, 0] = las.x
        pc_mmap[:, 1] = las.y
        pc_mmap[:, 2] = las.z
        pc_mmap[:, 3] = intensity
        # pc_mmap[:, 4] = las.classification
        # pc_mmap[:, 5] = red
        # pc_mmap[:, 6] = green
        # pc_mmap[:, 7] = blue

        # pc_mmap.flush()
        return pc_mmap, tmp_file, filename


def read_laz_file_chunked(file_path, memmap=True, chunk_size=10_000_000, fields=None):
    """Read a .laz or .las file in chunks and return an Nx8 array holding:

        x, y, z, intensity, classification, red, green, blue

    Parameters
    ----------
    file_path : str
        Path to the LAZ/LAS file.
    memmap : bool
        If True, use a numpy.memmap on disk.
    chunk_size : int
        Number of points per chunk.

    Returns
    -------
    (array, tmp_file, filename)
        array is either a numpy.ndarray or numpy.memmap
    """
    if memmap:
        tmp_file, filename = tempfile.mkstemp(dir=".")
        pc = las_to_memmap(
            file_path=file_path,
            output_path=filename,
            chunk_size=chunk_size,
            fields=fields,
        )
        return pc, tmp_file, filename

    pc = las_to_recarray(file_path, fields=fields)
    return pc, None, None


# -------------------------------------------------------------------------
# Point cloud writing
# -------------------------------------------------------------------------


def write_pointcloud(points, file_path, transform=None, save_labels=False):
    """Write a point cloud to a PCD file in XYZI format.

    Parameters
    ----------
    points : ndarray
        Typically shaped (H, W, 4) or (N, 4). Will be reshaped to (N, 4).

    file_path : str
        Output PCD filename.

    transform : tuple(position, rotation_matrix), optional
        If provided, applies the transform before writing.

    save_labels: bool If classification labels should be saved alongside the cloud
    """
    shape = points.shape
    if save_labels:
        points = points.reshape((-1, 5))
    else:
        points = points.reshape((-1, 4))

    if transform is not None:
        position, rotation = transform
        points[:, :3] = (rotation.T @ points[:, :3].T).T + position

    if save_labels:
        pc = PointCloud.from_xyzil_points(points)
    else:
        pc = PointCloud.from_xyzi_points(points)

    pc.metadata.height, pc.metadata.width = shape[:2]
    pc.save(file_path)


# -------------------------------------------------------------------------
# Trajectory reading
# -------------------------------------------------------------------------


def read_trajectory_file(file_path):
    """Read a trajectory file formatted as:

        timestamp  unused  y  x  z  roll  pitch  yaw

    The function returns tuples of (timestamp, position, orientation_vector).

    Parameters
    ----------
    file_path : str

    Returns
    -------
    list of tuple
        Each element is (timestamp, position(3,), orientation_rpy(3,)).
    """
    trajectory = []
    if file_path.endswith("dfnav"):
        return parse_nmea_file(file_path)
    if file_path.endswith("yaml"):
        return parse_pose_yaml(file_path)
    with open(file_path, "r") as f:
        header = True
        for line in f:
            if header:
                header = False
                continue

            parts = line.strip().split()
            timestamp = float(parts[0])

            # File order is X/Y swapped → convert to array in x,y,z order
            position = np.array(
                [
                    float(parts[2]),  # x
                    float(parts[1]),  # y
                    float(parts[3]),  # z
                ]
            )

            orientation = np.array(
                [
                    float(parts[4]),
                    float(parts[5]),
                    float(parts[6]),
                ]
            )

            trajectory.append((timestamp, position, orientation))

    return trajectory


# -------------------------------------------------------------------------
# Image saving
# -------------------------------------------------------------------------


def save_image(
    image, output_path, timestamp=None, position=None, multiplier=1, color=False
):
    """Save a range/intensity/RGB image and optionally embed minimal GPS EXIF.

    Parameters
    ----------
    image : ndarray
        Image data to save.

    output_path : str

    timestamp : float or None
        Optional timestamp embedded as EXIF DateTime.

    position : ndarray or None
        (x, y, z) in ETRS-TM35FIN coordinates. Converted to WGS84 EXIF GPS.

    multiplier : float
        Scaling factor applied to 16‑bit grayscale data.

    color : bool
        If True, image is assumed to already be 3‑channel RGB.
    """

    if not color:
        image = multiplier * image
        image[image < 0] = -1
        image = image.astype(np.uint16)

    cv2.imwrite(output_path, image)

    img = Image.open(output_path)
    exif = img.getexif()

    # Encode GPS EXIF if position given
    if position is not None:
        lat, lon = ETRSTM35FINxy_to_WGS84lalo(*position[:2])
        exif[ExifTags.Base.GPSInfo] = {
            GPS.GPSLatitudeRef: "N" if lat >= 0 else "S",
            GPS.GPSLatitude: abs(lat),
            GPS.GPSLongitudeRef: "E" if lon >= 0 else "W",
            GPS.GPSLongitude: abs(lon),
            GPS.GPSAltitudeRef: 0,
            GPS.GPSAltitude: float(position[2]),
        }

    if timestamp is not None:
        exif[ExifTags.Base.DateTime] = str(timestamp)
    # img.save(output_path, exif=exif)


# -------------------------------------------------------------------------
# Classification pseudo‑color export
# -------------------------------------------------------------------------


def save_classification_image(
    image, output_path, timestamp=None, position=None, color=False
):
    """Save a classification image. When color=True, a tab20 colormap is applied.

    Parameters
    ----------
    image : ndarray
        Integer classification values (0–255).

    output_path : str

    color : bool
        If True, apply matplotlib colormap.

    timestamp, position :
        Passed to EXIF generation (if color=False).
    """
    logging.debug(("CLASSIFICATION IMAGE", output_path))

    if color:
        colors = np.asarray(mpl.color_sequences["tab20b"])
        color_img = np.zeros((*image.shape, 3), dtype=np.uint8)

        for idx, _ in np.ndenumerate(image):
            cid = image[idx] % len(colors)
            color_img[idx] = (colors[cid] * 255).astype(np.uint8)

        cv2.imwrite(output_path, color_img)
        return

    cv2.imwrite(output_path, image)


def match_las_and_trajectory(las_dir, traj_dir=None, sort=False):
    """Match LAS/LAZ point cloud files with trajectory text files by timestamp
    pattern appearing in both filenames.

    Parameters
    ----------
    las_dir : str
        Directory containing LAS/LAZ files, or single LAS/LAZ file.
    traj_dir : str
        Directory containing trajectory TXT files.
    sort : bool
        Sort LAS files alphabetically.

    Returns
    -------
    list[tuple(str, str)]
        List of (las_file, traj_file) pairs.
    """

    # Single-file shortcut
    if os.path.isfile(las_dir):
        try:
            return [(os.path.basename(las_dir), os.path.basename(traj_dir))]
        except FileNotFoundError:
            return ((os.path.basename(las_dir),),)

    las_files = [
        f for f in os.listdir(las_dir) if f.endswith(".las") or f.endswith(".laz")
    ]

    if sort:
        las_files.sort()
    if traj_dir is None:
        # return tuple of tuples
        return ((las_file,) for las_file in las_files)

    traj_files = [
        f for f in os.listdir(traj_dir) if f.endswith(".txt") and f.startswith("Traj")
    ]

    # One LAS → one trajectory
    if len(las_files) == 1:
        return (las_files[0], traj_files[0])

    pairs = []

    # Extract identifier: 'RecordXXXX_YYYYMMDD_HHMMSS'
    sample = las_files[0]
    id_start = sample.find("Record") + 10
    id_end = id_start + 13

    for lasf in las_files:
        ident = lasf[id_start:id_end]
        for trajf in traj_files:
            if ident in trajf:
                pairs.append((lasf, trajf))
                traj_files.remove(trajf)
                break

    return pairs


def update_classification(point_cloud, orginal_file, target_path):
    las = laspy.read(orginal_file)
    classification = point_cloud[:, 4].astype(np.uint8)
    classification[classification >= 32] = 0
    las.classification = classification  # point_cloud[:, 4]
    laspy.convert(las, file_version="1.4")

    las.add_extra_dim(
        laspy.ExtraBytesParams(
            name="id", type=np.uint32, description="Instance id from coco"
        )
    )
    ids = point_cloud[:, -1].astype(np.uint32)
    las.id = ids
    las.write(target_path)


def create_progress_pountclouds(point_cloud, orginal_file, target_path):
    las = laspy.read(orginal_file)
    laspy.convert(las, file_version="1.4")

    dims_to_add = []
    for i in range(point_cloud.shape[1] - 3):
        dims_to_add.append(
            laspy.ExtraBytesParams(
                name=f"i_{i}", type=np.uint32, description=f"expansions{i}"
            )
        )
    las.add_extra_dims(dims_to_add)

    for i in range(point_cloud.shape[1] - 3):
        ids = point_cloud[:, i + 3].astype(np.uint32)
        las.__setattr__(f"i_{i}", ids)

    las.write(target_path)
