import openeo
from openeo.processes import ProcessBuilder, array_modify, normalized_difference, drop_dimension, quantiles, sd, mean, array_apply, array_concat

# percentiles p10, p50, p90
# std dev
# tsteps ts0, ts1, ts5

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

def callback(x: ProcessBuilder, index, lenx, nlenx):
	return array_modify(data=x, values=indices[index](*[x.array_element(i) for i in range(lenx)]), index=nlenx)

def compute_features(datacube, features:list):
	dc = datacube
	lenx = len(datacube.metadata._band_dimension.bands)
	nlenx = lenx
	for i in features:
		dc = dc.apply_dimension(dimension="bands", process=lambda x: callback(x,i,lenx,nlenx))
		nlenx += 1
	return dc
