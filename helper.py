"""
Module for calculating a list of vegetation indices from a datacube containing bands without a user having to implement callback functions
"""

from openeo import Connection
from openeo.rest.datacube import DataCube
from openeo.processes import ProcessBuilder, array_modify, array_concat, array_apply, power, sqrt, clip, if_, multiply, divide, arccos, linear_scale_range, add, subtract

import numpy as np


WL_B04 = 0.6646
WL_B08 = 0.8328
WL_B11 = 1.610
one_over_pi = 1. / np.pi

# source: https://git.vito.be/projects/LCLU/repos/satio/browse/satio/rsindices.py
ndvi = lambda B04, B08: (B08 - B04) / (B08 + B04)
ndmi = lambda B08, B11: (B08 - B11) / (B08 + B11)
ndgi = lambda B03, B04: (B03 - B04) / (B03 + B04)


def anir(B04, B08, B11):
    a = sqrt(np.square(WL_B08 - WL_B04) + power(B08 - B04, 2))
    b = sqrt(np.square(WL_B11 - WL_B08) + power(B11 - B08, 2))
    c = sqrt(np.square(WL_B11 - WL_B04) + power(B11 - B04, 2))

    # calculate angle with NIR as reference (ANIR)
    site_length = (power(a, 2) + power(b, 2) - power(c, 2)) / (2 * a * b)
    site_length = if_(site_length.lt(-1), -1, site_length)
    site_length = if_(site_length.gt(1), 1, site_length)

    return multiply(one_over_pi, arccos(site_length))


ndre1 = lambda B05, B08: (B08 - B05) / (B08 + B05)
ndre2 = lambda B06, B08: (B08 - B06) / (B08 + B06)
ndre5 = lambda B05, B07: (B07 - B05) / (B07 + B05)

indices = {
    "NDVI": [ndvi, (0,1)],
    "NDMI": [ndmi, (-1,1)],
    "NDGI": [ndgi, (-1,1)],
    "ANIR": [anir, (0,1)],
    "NDRE1": [ndre1, (-1,1)],
    "NDRE2": [ndre2, (-1,1)],
    "NDRE5": [ndre5, (-1,1)]
}


def _callback(x: ProcessBuilder, index_list: list, datacube: DataCube, scaling_factor: int, to_scale=True) -> ProcessBuilder:
    lenx = len(datacube.metadata._band_dimension.bands)
    tot = x
    for idx in index_list:
        if idx not in indices.keys(): raise NotImplementedError("Index " + idx + " has not been implemented.")
        band_indices = [datacube.metadata.get_band_index(band) for band in
                        indices[idx][0].__code__.co_varnames[:indices[idx][0].__code__.co_argcount]]
        lenx += 1
        if to_scale:
            tot = array_modify(data=tot, values=lin_scale_range(indices[idx][0](*[tot.array_element(i) for i in band_indices]),*indices[idx][1],0,scaling_factor), index=lenx)
        else:
            tot = array_modify(data=tot, values=indices[idx][0](*[tot.array_element(i) for i in band_indices]), index=lenx)
    return tot


def compute_indices(datacube: DataCube, index_list: list, scaling_factor: int) -> DataCube:
    """
    Computes a list of indices from a datacube

    param datacube: an instance of openeo.rest.DataCube
    param index_list: a list of indices. The following indices are currently implemented: NDVI, NDMI, NDGI, ANIR, NDRE1, NDRE2 and NDRE5
    return: the datacube with the indices attached as bands

    """
    return datacube.apply_dimension(dimension="bands",
                                    process=lambda x: _callback(x, index_list, datacube, scaling_factor, to_scale=False)).rename_labels('bands',
                                                                                                        target=datacube.metadata.band_names + index_list)


def lin_scale_range(x,inputMin,inputMax,outputMin,outputMax):
    return add(multiply(divide(subtract(x,inputMin), subtract(inputMax, inputMin)), subtract(outputMax, outputMin)), outputMin)



