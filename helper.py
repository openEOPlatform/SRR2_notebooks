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
ndvi = lambda b4,b8: (b8 - b4) / (b8 + b4) 
ndmi = lambda b8,b11: (b8 - b11) / (b8 + b11)
ndgi = lambda b3,b4: (b3 - b4) / (b3 + b4)
def anir(b4, b8, b11):
    a = np.sqrt(np.square(WL_B08 - WL_B04) + np.square(b8 - b4))
    b = np.sqrt(np.square(WL_B11 - WL_B08) + np.square(b11 - b8))
    c = np.sqrt(np.square(WL_B11 - WL_B04) + np.square(b11 - b4))

    # calculate angle with NIR as reference (ANIR)
    site_length = (np.square(a) + np.square(b) - np.square(c)) / (2 * a * b)
    site_length[site_length < -1] = -1
    site_length[site_length > 1] = 1

    return 1. / np.pi * np.arccos(site_length)
ndre1 = lambda b5,b8: (b8 - b5) / (b8 + b5)
ndre2 = lambda b6,b8: (b8 - b6) / (b8 + b6)
ndre5 = lambda b5,b7: (b7 - b5) / (b7 + b5)

indices = {
	"NDVI": ndvi,
	"NDMI": ndmi,
	"NDGI": ndgi,
	"ANIR": anir,
	"NDRE1": ndre1,
	"NDRE2": ndre2,
	"NDRE5": ndre5
}

def _callback(x: ProcessBuilder, index_list:list, lenx:int) -> ProcessBuilder:
	tot = x
	nlenx = lenx
	for i in index_list:
		if i not in indices.keys(): raise NotImplementedError("Index "+i+" has not been implemented.")
		nlenx += 1
		tot = array_modify(data=tot,values=indices[i](*[tot.array_element(i) for i in range(lenx)]),index=nlenx)
	return tot


def compute_indices(datacube:DataCube, index_list:list) -> DataCube:
	"""
	Computes a list of indices from a datacube
	
	param datacube: an instance of openeo.rest.DataCube
	param index_list: a list of indices. The following indices are currently implemented: NDVI, NDMI, NDGI, ANIR, NDRE1, NDRE2 and NDRE5
	return: the datacube with the indices attached as bands

	"""
	lenx = len(datacube.metadata._band_dimension.bands)
	return datacube.apply_dimension(dimension="bands", process=lambda x: _callback(x,index_list,lenx))
