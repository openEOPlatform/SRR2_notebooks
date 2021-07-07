"""
Module for calculating a list of vegetation indices from a datacube containing bands without a user having to implement callback functions
"""

import openeo
from openeo.rest.datacube import DataCube
from openeo.processes import ProcessBuilder, array_modify, array_concat
import numpy as np

WL_B04 = 0.6646
WL_B08 = 0.8328
WL_B11 = 1.610

# source: https://git.vito.be/projects/LCLU/repos/satio/browse/satio/rsindices.py
ndvi = lambda B04,B08: (B08 - B04) / (B08 + B04) 
ndmi = lambda B08,B11: (B08 - B11) / (B08 + B11)
ndgi = lambda B03,B04: (B03 - B04) / (B03 + B04)
def anir(B04, B08, B11):
    a = np.sqrt(np.square(WL_B08 - WL_B04) + np.square(B08 - B04))
    b = np.sqrt(np.square(WL_B11 - WL_B08) + np.square(B11 - B08))
    c = np.sqrt(np.square(WL_B11 - WL_B04) + np.square(B11 - B04))

    # calculate angle with NIR as reference (ANIR)
    site_length = (np.square(a) + np.square(b) - np.square(c)) / (2 * a * b)
    site_length[site_length < -1] = -1
    site_length[site_length > 1] = 1

    return 1. / np.pi * np.arccos(site_length)
ndre1 = lambda B05,B08: (B08 - B05) / (B08 + B05)
ndre2 = lambda B06,B08: (B08 - B06) / (B08 + B06)
ndre5 = lambda B05,B07: (B07 - B05) / (B07 + B05)

indices = {
	"NDVI": ndvi,
	"NDMI": ndmi,
	"NDGI": ndgi,
	"ANIR": anir,
	"NDRE1": ndre1,
	"NDRE2": ndre2,
	"NDRE5": ndre5
}

def _callback(x: ProcessBuilder, index_list:list, datacube:DataCube) -> ProcessBuilder:
	lenx = len(datacube.metadata._band_dimension.bands)
	tot = x
	for idx in index_list:
		if idx not in indices.keys(): raise NotImplementedError("Index "+idx+" has not been implemented.")
		band_indices = [datacube.metadata.get_band_index(band) for band in indices[idx].__code__.co_varnames]
		lenx += 1
		tot = array_modify(data=tot,values=indices[idx](*[tot.array_element(i) for i in band_indices]),index=lenx)
	return tot


def compute_indices(datacube:DataCube, index_list:list) -> DataCube:
	"""
	Computes a list of indices from a datacube
	
	param datacube: an instance of openeo.rest.DataCube
	param index_list: a list of indices. The following indices are currently implemented: NDVI, NDMI, NDGI, ANIR, NDRE1, NDRE2 and NDRE5
	return: the datacube with the indices attached as bands

	"""
	return datacube.apply_dimension(dimension="bands", process=lambda x: _callback(x,index_list,datacube)).rename_labels('bands', target=datacube.metadata.band_names+index_list)
