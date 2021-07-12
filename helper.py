"""
Module for calculating a list of vegetation indices from a datacube containing bands without a user having to implement callback functions
"""

from openeo import Connection
from openeo.rest.datacube import DataCube
from openeo.processes import ProcessBuilder, array_modify, power, sqrt, if_, multiply, arccos
import numpy as np
import pytest, rasterio


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
    "NDVI": ndvi,
    "NDMI": ndmi,
    "NDGI": ndgi,
    "ANIR": anir,
    "NDRE1": ndre1,
    "NDRE2": ndre2,
    "NDRE5": ndre5
}


def _callback(x: ProcessBuilder, index_list: list, datacube: DataCube) -> ProcessBuilder:
    lenx = len(datacube.metadata._band_dimension.bands)
    tot = x
    for idx in index_list:
        if idx not in indices.keys(): raise NotImplementedError("Index " + idx + " has not been implemented.")
        band_indices = [datacube.metadata.get_band_index(band) for band in
                        indices[idx].__code__.co_varnames[:indices[idx].__code__.co_argcount]]
        lenx += 1
        tot = array_modify(data=tot, values=indices[idx](*[tot.array_element(i) for i in band_indices]), index=lenx)
    return tot


def compute_indices(datacube: DataCube, index_list: list) -> DataCube:
    """
    Computes a list of indices from a datacube

    param datacube: an instance of openeo.rest.DataCube
    param index_list: a list of indices. The following indices are currently implemented: NDVI, NDMI, NDGI, ANIR, NDRE1, NDRE2 and NDRE5
    return: the datacube with the indices attached as bands

    """
    return datacube.apply_dimension(dimension="bands",
                                    process=lambda x: _callback(x, index_list, datacube)).rename_labels('bands',
                                                                                                        target=datacube.metadata.band_names + index_list)





@pytest.mark.parametrize(["index", "bands", "expected"], [
    ("NDVI", ["B04", "B08"], [[0.8563234, 0.845991], [0.82082057, 0.841846]]),
    ("NDMI", ["B08", "B11"], [[0.48937103, 0.4992478], [0.47535053, 0.50764006]]),
])
def test_vegindex_calculator_ndvi(auth_connection: Connection, index, bands, expected):
    x = 640860.000
    y = 5676170.000
    bbox = (x,y,x+20,y+20)

    s2 = auth_connection.load_collection("TERRASCOPE_S2_TOC_V2",
                                    spatial_extent={'west':bbox[0],'east':bbox[2],'south':bbox[1],'north':bbox[3], 'crs':"EPSG:32631"},
                                    temporal_extent=["2018-01-21", "2018-01-21"],
                                    bands=bands+["SCL"])

    s2_masked = s2.process("mask_scl_dilation", data=s2, scl_band_name="SCL").filter_bands(bands)
    feats = compute_indices(s2_masked, [index])
    feats.download("./tmp/feats.tif", format="GTiff")

    with rasterio.open('./tmp/feats.tif') as dataset:
        result = dataset.read(3)

    np.testing.assert_almost_equal(actual=result, desired=expected, decimal=7)