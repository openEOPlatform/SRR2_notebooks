import pytest, rasterio

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
    feats = compute_indices(s2_masked, [index], 1)
    feats.download("./data/feats.tif", format="GTiff")

    with rasterio.open('./data/feats.tif') as dataset:
        result = dataset.read(3)

    np.testing.assert_almost_equal(actual=result, desired=expected, decimal=7)