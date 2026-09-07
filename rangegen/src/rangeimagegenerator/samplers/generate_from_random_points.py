import numpy as np
from scipy.spatial.transform import Rotation as R

from rangeimagegenerator import Pose
# -------------------------------------------------------------------------
# Sampling helpers
# -------------------------------------------------------------------------


def calculate_sampling_range_average(minimum, maximum, avg, segment):
    span = avg
    cutoff_x_min = segment[0] * span
    cutoff_x_max = (1 - segment[1]) * span
    return minimum + cutoff_x_min, maximum - cutoff_x_max


def array_in_bounds(points, bounds):
    return (
        (points[:, 0] >= bounds[0][0])
        & (points[:, 0] <= bounds[0][1])
        & (points[:, 1] >= bounds[1][0])
        & (points[:, 1] <= bounds[1][1])
        & (points[:, 2] >= bounds[2][0])
        & (points[:, 2] <= bounds[2][1])
    )


def calculate_random_sampling_points(
    points,
    x_segment,
    y_segment,
    z_segment,
    number_of_sampling_points,
    sensor_orientation=None,
    classes=None,
):
    minimums = np.amin(points[:, :3], axis=0)
    maximums = np.amax(points[:, :3], axis=0)
    avgs = np.average(points[:, :3] - minimums, axis=0)
    sensor_orientation = (
        R.from_euler("xyz", sensor_orientation).as_quat()
        if sensor_orientation is not None
        else sensor_orientation
    )

    x_range = calculate_sampling_range_average(
        minimums[0], maximums[0], avgs[0], x_segment
    )
    y_range = calculate_sampling_range_average(
        minimums[1], maximums[1], avgs[1], y_segment
    )
    z_range = calculate_sampling_range_average(
        minimums[2], maximums[2], avgs[2], z_segment
    )

    bounds = (x_range, y_range, z_range)

    pts = points.copy()

    # if we are intrested in only some classes
    if classes is not None:
        pts = pts[np.isin(pts[:, 4], classes)]

    sampling_points = pts[array_in_bounds(pts[:, :3], bounds)]
    rng = np.random.default_rng()
    orientation = sensor_orientation if sensor_orientation is not None else [0, 0, 0]
    traj_len_str = len(str(number_of_sampling_points))
    poses = (
        Pose(position=rng.choice(sampling_points)[:3], orientation=orientation)
        for i in range(number_of_sampling_points)
    )
    base_ids = (str(i).zfill(traj_len_str) for i in range(number_of_sampling_points))
    return poses, base_ids


def random_sampling(points, cfg, classes=None):
    x_segment = cfg["processing"].get("x_segment", (0, 1))
    y_segment = cfg["processing"].get("y_segment", (0, 1))
    z_segment = cfg["processing"].get("z_segment", (0, 1))
    number_of_sampling_points = cfg["processing"].get("sampling_points", 25)
    sensor_orientation = cfg["sensor"].get("sensor_orientation", [0, 0, 0])

    return calculate_random_sampling_points(
        points,
        x_segment,
        y_segment,
        z_segment,
        number_of_sampling_points,
        sensor_orientation=sensor_orientation,
        classes=classes,
    )
