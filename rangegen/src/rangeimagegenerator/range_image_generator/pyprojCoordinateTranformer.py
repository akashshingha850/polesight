#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Fri Aug 27 12:36:32 2021

@author: timo
"""

from pyproj import Transformer
import logging
# ￼epsg.io to search coordinate systems
# 4326 is WGS84 ie. GPS position
# 3067 is ETHR89 ie metric coordinates in finlans

transformerWGS_84_to_ETRS89_TM35FIN_E_N = Transformer.from_crs(4326, 3067)
transformerETRS89_TM35FIN_E_N_to_WGS_84 = Transformer.from_crs(3067, 4326)

transformerWGS_84_to_UTM35 = Transformer.from_crs(4326, 32633)
# 32633
# 32635
# +proj=utm +zone=35 +datum=WGS84 +units=m +no_defs


def WGS84lalo_to_ETRSTM35FINxy(lat, lon):
    # X = EASTING , y = NORHING
    return transformerWGS_84_to_ETRS89_TM35FIN_E_N.transform(lat, lon)


def WGS84lalo_to_UTM35(lat, lon):
    return transformerWGS_84_to_UTM35.transform(lat, lon)


def ETRSTM35FINxy_to_WGS84lalo(easting, northing):
    return transformerETRS89_TM35FIN_E_N_to_WGS_84.transform(easting, northing)


from pyproj import CRS, Transformer

# --- Define CRS ---
wgs84 = CRS.from_epsg(4326)

custom_tm = CRS.from_proj4(
    "+proj=tmerc +lat_0=0 +lon_0=29 +k=1 "
    "+x_0=500000 +y_0=0 +ellps=GRS80 +units=m +no_defs"
)

# Transformers
to_tm = Transformer.from_crs(wgs84, custom_tm, always_xy=True)
to_wgs = Transformer.from_crs(custom_tm, wgs84, always_xy=True)


# --- NMEA parsing ---
def nmea_to_decimal(coord, direction):
    # coord format: DDMM.MMMMM
    deg = int(coord[: 2 if direction in ["N", "S"] else 3])
    minutes = float(coord[2 if direction in ["N", "S"] else 3 :])
    value = deg + minutes / 60.0
    return -value if direction in ["S", "W"] else value


def parse_gga(line):
    parts = line.split(",")
    lat = nmea_to_decimal(parts[2], parts[3])
    lon = nmea_to_decimal(parts[4], parts[5])
    alt = float(parts[9])
    return lat, lon, alt


if __name__ == "__main__":
    # ETRS-TM35FIN: 7203406  431820
    # E: 431830  N: 7203400
    # lat (y suunta): 64.---  lon (x suunta): 25.000
    print(
        WGS84lalo_to_ETRSTM35FINxy(64.86741742, 25.02979013), "=", (406643, 7195132)
    )  # Should be same
    logging.info((WGS84lalo_to_ETRSTM35FINxy(65.06433854467119, 25.461628221645658)))

    logging.info((ETRSTM35FINxy_to_WGS84lalo(406643, 7195132)))

    logging.info((ETRSTM35FINxy_to_WGS84lalo(427635.6580219937, 7216506.006702797)))
