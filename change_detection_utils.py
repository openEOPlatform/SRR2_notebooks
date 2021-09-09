from openeo import Connection
from openeo.rest.datacube import DataCube, PGNode, THIS
from openeo.rest.job import RESTJob
from openeo.processes import *
import numpy as np
import math
import xarray as xr

def fit_function_season(x:ProcessBuilder,parameters):
    pi=math.pi
    a0 = array_element(parameters,0)
    a1 = array_element(parameters,1)
    a2 = array_element(parameters,2)
    return a0 +a1*cos(2*pi/31557600*x) + a2* sin(2*pi/31557600*x)

def seasonal_curve_fitting(datacube: DataCube) -> DataCube:
    args_fit_curve= {
        "data": THIS,
        "parameters": [1,1,1], # Initial guess of the parameters
        "dimension": 't',      # Fit the function along the temporal dimension
        "function": datacube._get_callback(fit_function_season, parent_parameters=["data","parameters"])}

    return datacube.process("fit_curve",args_fit_curve)


def seasonal_curve_predicting(datacube: DataCube,parameters: PGNode) -> DataCube:
    args_predict_curve= {
        "data": THIS,
        "parameters": parameters,
        "dimension": 't',
        "function": datacube._get_callback(fit_function_season, parent_parameters=["data","parameters"])}

    return datacube.process("predict_curve",args_predict_curve)

def compute_residual(datacube: DataCube,bands: list) -> DataCube:
    for i,b in enumerate(bands):
        if i==0:
            rmse = datacube.band(b)**2
        else:
            rmse = datacube.band(b)**2 + rmse
    return (rmse/len(bands))**0.5

def get_bbox_from_job(job: RESTJob) -> dict:
    spatial_extent = None
    job_descr = job.describe_job()
    for n in job_descr['process']['process_graph']:
        if job_descr['process']['process_graph'][n]['process_id'] == "load_collection":
            spatial_extent = job_descr['process']['process_graph'][n]['arguments']['spatial_extent']
    return spatial_extent

def plot_detected_changes(netcdfPath='result.nc',monthlyAggregate=True,subsequentAlarms=3,backgroundTiles='OSM'):
    import xarray as xr
    import geoviews as gv
    import geoviews.feature as gf
    from cartopy import crs
    
    alarms = xr.open_dataarray(netcdfPath)
    ## Apply filtering using rolling window
    alarms = alarms.rolling(t=subsequentAlarms, center=False).mean().fillna(0)
    alarms = alarms.where(alarms>0.9).clip(0,0.5)+0.5
    alarms.name = 'Detected_changes'

    timeDim = 't'
    
    if monthlyAggregate:
        alarms = alarms.groupby('t.month').max('t')
        timeDim = 'month'
    
    tiles = gv.tile_sources.OSM
    if backgroundTiles == 'ESRI':
        tiles = gv.tile_sources.EsriImagery
    try:
        in_epsg = alarms.spatial_ref.item(0)
    except:
        try:
            in_epsg = alarms.crs.item(0)
        except:
            raise Exception("Projection not found in the input file!")
    dataset = gv.Dataset(alarms, ['x', 'y', timeDim], crs=crs.epsg(in_epsg))
    images = dataset.to(gv.Image, ['x', 'y'], 'Detected_changes', timeDim)
    return images.opts(cmap='Reds', width=600, height=500) * tiles

def apply_gt0(x:ProcessBuilder):
    return x.gt(0)

def apply_clipping(x:ProcessBuilder):
    return clip(x,0,5000)

def download_raw_and_predicted(connection,collection,point_coords,temporal_extent,temporal_extent_reference,band,filename):
    '''
    Loads the required L2A band, applies the cloud mask, computes the coefficients of the fitted seasonal function and
    returns a datacube with the Sentinel-2 pre-processed timeseries and the predicted timeseries following the fitted model.
    '''
    # Use the point coordinates in input to define a small bounding box adding a small delta
    spatial_extent  = {'west':point_coords[0]-0.0001,'east':point_coords[0]+0.0001,'south':point_coords[1]-0.0001,'north':point_coords[1]+0.0001}
    # Load the required S2 band
    l2a_bands = connection.load_collection(collection,spatial_extent=spatial_extent,bands=[band],temporal_extent=temporal_extent)
    # Load the cloud mask and apply it
    cloud_band      = connection.load_collection('s2cloudless_alps',spatial_extent=spatial_extent,temporal_extent=temporal_extent)
    args_resample_cube_temporal= {"data": THIS,"method": "nearest","target": l2a_bands}
    cloud_band_resampled = cloud_band.process("resample_cube_temporal",args_resample_cube_temporal)
    cloud_band_resampled = cloud_band_resampled.apply(apply_gt0)
    l2a_bands_masked = l2a_bands.mask(mask=cloud_band_resampled,replacement=0)
    # Clip the data to avoid unmasked clouds
    l2a_bands_masked_clipped = l2a_bands_masked.apply(apply_clipping)
    # Apply seasonal curve fitting
    curve_fitting = seasonal_curve_fitting(l2a_bands_masked_clipped.filter_temporal(temporal_extent_reference))
    # Compute the values 
    curve_prediction = seasonal_curve_predicting(l2a_bands_masked_clipped,curve_fitting).rename_labels(target=[band+"_predicted"],dimension="bands")
    # Merge real and predicted data and store it in a netCDF and download it
    l2a_and_predicted = l2a_bands_masked_clipped.merge_cubes(curve_prediction)
    l2a_and_predicted_nc = l2a_and_predicted.save_result(format="NetCDF")
    l2a_and_predicted_nc.download(filename)
    return

def download_rgb_and_predicted(connection,collection,spatial_extent,temporal_extent,bands,filename,jobIdFitting):
    '''
    1. Loads the required L2A bands, computes the median over the time range and stores the result as the first PNG.
    2. Loads the fitted parameters and use them to predict the values for the same tim range, take the median of the result and save it as a PNG.
    '''
    # Load the required S2 band and s2cloudless
    l2a_bands = connection.load_collection(collection,spatial_extent=spatial_extent,bands=bands,temporal_extent=temporal_extent)
    cloud_band      = connection.load_collection('s2cloudless_alps',spatial_extent=spatial_extent,temporal_extent=temporal_extent)
    args_resample_cube_temporal= {"data": THIS,"method": "nearest","target": l2a_bands}
    cloud_band_resampled = cloud_band.process("resample_cube_temporal",args_resample_cube_temporal)
    cloud_band_resampled = cloud_band_resampled.apply(apply_gt0)
    l2a_bands_masked = l2a_bands.mask(mask=cloud_band_resampled,replacement=0)
    l2a_bands_masked_clipped = l2a_bands_masked.apply(apply_clipping)
    # Take the median over time
    l2a_median = l2a_bands_masked_clipped.reduce_dimension(dimension="DATE",reducer="median")
    # Scale the values to 0-255
    lin_scale = PGNode("linear_scale_range", arguments={"x": {"from_parameter": "x"},"inputMin": 0, "inputMax": 1800, "outputMin": 0, "outputMax": 255})
    l2a_median_255 = l2a_median.apply(lin_scale)
    # Save the result to PNG and download
    l2a_median_255_PNG = l2a_median_255.save_result(format="PNG")
    l2a_median_255_PNG.download(filename)
    
    curve_fitting_loaded = PGNode("load_result", id=jobIdFitting)
    from change_detection_utils import seasonal_curve_predicting
    curve_prediction = seasonal_curve_predicting(l2a_bands,curve_fitting_loaded)
    l2a_median_pred = curve_prediction.reduce_dimension(dimension="DATE",reducer="median")
    l2a_median_pred_255 = l2a_median_pred.apply(lin_scale)
    l2a_median_pred_255_PNG = l2a_median_pred_255.save_result(format="PNG")
    l2a_median_pred_255_PNG.download(filename.split('.')[0]+"_pred.png")
    return

def download_S1_raw_and_predicted(connection,collectionId,point_coords,temporal_extent,temporal_extent_reference,band,filename):
    '''
    Loads the required S1 band, computes the coefficients of the fitted seasonal function and returns
    a datacube with the Sentinel-1 pre-processed timeseries and the predicted timeseries following the fitted model.
    '''
    # Use the point coordinates in input to define a small bounding box adding a small delta
    spatial_extent  = {'west':point_coords[0]-0.0001,'east':point_coords[0]+0.0001,'south':point_coords[1]-0.0001,'north':point_coords[1]+0.0001}
    # Load the required S1 band
    collection      = 'SAR2Cube_L0_117_ASC_ST_2016_2020_IFG_LIA_DEM'
    sar_data = conn.load_collection(collection)
    intensity_data = sar_data.process("load_result", id=collectionId)
    intensity_data_temp = intensity_data.filter_temporal(temporal_extent)
    intensity_data_bbox = intensity_data_temp.filter_bbox(spatial_extent)
    intensity_data_band = intensity_data_bbox.rename_labels("bands",["VV","VH"]).filter_bands(band)
#     Apply seasonal curve fitting
    curve_fitting = seasonal_curve_fitting(intensity_data_band.filter_temporal(temporal_extent_reference))
#     Compute the values 
    curve_prediction = seasonal_curve_predicting(intensity_data_band,curve_fitting).rename_labels(target=[band+"_predicted"],dimension="bands")
#     Merge real and predicted data and store it in a netCDF and download it
    intensity_data_pred = intensity_data_band.merge_cubes(curve_prediction)
    intensity_data_nc = intensity_data_pred.save_result(format="NetCDF")
    intensity_data_nc.download(filename)
    return