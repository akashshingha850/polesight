from datetime import datetime
import csv

import yaml
import numpy as np

import logging

from .pyprojCoordinateTranformer import (
    WGS84lalo_to_ETRSTM35FINxy,
)


def dm_to_dd(dm, direction):
    """Convert NMEA degree-minute format to decimal degrees."""
    if not dm:
        return None

    if len(dm.split(".")[0]) > 4:
        deg_len = 3  # longitude
    else:
        deg_len = 2  # latitude

    degrees = int(dm[:deg_len])
    minutes = float(dm[deg_len:])
    dd = degrees + (minutes / 60)

    if direction in ["S", "W"]:
        dd *= -1

    return dd


def parse_nmea_file(filepath):
    data = []
    initial_pos = None
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        current_date = None

        for line in f:
            line = line.strip()

            if line.startswith("$GNRMC"):
                parts = line.split(",")

                time_raw = parts[1]
                date_raw = parts[9]

                if time_raw and date_raw:
                    current_date = datetime.strptime(
                        date_raw + time_raw[:6], "%d%m%y%H%M%S"
                    )
                continue
                lat = dm_to_dd(parts[3], parts[4])
                lon = dm_to_dd(parts[5], parts[6])

                data.append(
                    {
                        "timestamp": current_date,
                        "latitude": lat,
                        "longitude": lon,
                        "altitude_m": None,
                        "source": "RMC",
                    }
                )

            elif line.startswith("$GNGGA"):
                parts = line.split(",")

                if current_date is None:
                    continue

                time_raw = parts[1]

                dt = datetime.strptime(
                    current_date.strftime("%Y%m%d") + time_raw[:6], "%Y%m%d%H%M%S"
                )

                lat = dm_to_dd(parts[2], parts[3])
                lon = dm_to_dd(parts[4], parts[5])
                # OFFSETS IN LIDAR DATA: 333816.611, 7222578.964, 35.087
                x, y = WGS84lalo_to_ETRSTM35FINxy(lat, lon)

                # x,y = to_tm.transform(lon,lat )

                # x= x
                # y= -y

                fix_quality = int(parts[6]) if parts[6] else None
                altitude = float(parts[9]) if parts[9] else None

                if initial_pos is None:
                    initial_pos = np.array([x, y, altitude + 20.087])

                pos = np.array([x, y, altitude + 18])  # - initial_pos

                data.append([float(time_raw), pos, (0, 0, 0)])

                # data.append({
                #     "timestamp": dt,
                #     "latitude": lat,
                #     "longitude": lon,
                #     "altitude_m": altitude,
                #     "fix_quality": fix_quality,
                #     "source": "GGA"
                # })

    return data


def parse_pose_yaml(filepath):
    with open(filepath) as f:
        pose_list = yaml.safe_load(f)["poseList"]

    return [
        [
            v[0],
            np.array(
                [
                    v[1],
                    v[2],
                    v[3],
                ]
            ),
            [0, 0, 0],
        ]
        for k, v in pose_list.items()
        if k != "poseListSize"
    ]


def save_csv(data, filename="parsed_rtk_with_height.csv"):
    keys = data[0].keys()

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    file_path = "20260526-100457_Rtk (copy).txt"

    parsed = parse_nmea_file(file_path)

    for row in parsed[:]:
        # if row["source"] == "GGA":
        logging.debug((row))

    # save_csv(parsed)
