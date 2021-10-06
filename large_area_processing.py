import openeo
import geopandas as gpd
import pandas as pd
import time
from helper import compute_indices, lin_scale_range
from openeo.processes import ProcessBuilder, if_, array_concat
from openeo.util import deep_get

temporal_partition_options = {
        "indexreduction": 0,
        "temporalresolution": "None",
        "tilesize": 256
    }

col_palette = {0: (0.9450980392156862, 0.403921568627451, 0.27058823529411763, 1.0), 
            1: (1.0, 0.7764705882352941, 0.36470588235294116, 1.0), 
            2: (0.4823529411764706, 0.7843137254901961, 0.6431372549019608, 1.0), 
            3: (0.2980392156862745, 0.7647058823529411, 0.8509803921568627, 1.0), 
            4: (0.5764705882352941, 0.39215686274509803, 0.5529411764705883, 1.0), 
            99: (0.25098039215686274, 0.25098039215686274, 0.25098039215686274, 1.0)}

udf_rf = """
from typing import Dict
from openeo_udf.api.datacube import DataCube
import pickle
import urllib.request
import xarray
import sklearn
from openeo.udf.xarraydatacube import XarrayDataCube
import functools

@functools.lru_cache(maxsize=25)
def load_model():
    return pickle.load(urllib.request.urlopen("https://artifactory.vgt.vito.be:443/auxdata-public/openeo/rf_model_s2only.pkl"))

def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:
    array = cube.get_array()
    stacked_array = array.stack(pixel=("x","y"))
    stacked_array = stacked_array.transpose()
    stacked_array_filtered = stacked_array[~np.isnan(stacked_array).any(axis=1)]
    if len(stacked_array_filtered) == 0:
        result = np.full(np.multiply(*array.shape[1:]),np.nan)
    else:
        clf = load_model()
        probs = clf.predict_proba(stacked_array_filtered)
        pred_array = np.argmax(probs,axis=1)
        none_indices = np.where(np.amax(probs,axis=1)<0.5)
        pred_array[none_indices] = 99
        result = np.full(len(stacked_array),np.nan)
        result[~np.isnan(stacked_array).any(axis=1)] = pred_array
    da = xarray.DataArray(np.transpose(result.reshape(array.shape[1:])),coords=[array.coords["x"],array.coords["y"]], dims=["x","y"])
    return DataCube(da)
"""

def read_or_create_csv(grid, index):
    try:
        status_df = pd.read_csv(csv_path.format(index), index_col=0)
    except FileNotFoundError:
        status_df = pd.DataFrame(columns=["name", "status", "id", "cpu", "memory", "duration"])

        for i in range(len(grid)):
            status_df = status_df.append(
                {"name": grid.name[i], "status": "pending", "id": None, "cpu": None, "memory": None, "duration": None}, ignore_index=True)

        status_df.to_csv(csv_path.format(index))

    return status_df

def computeStats(input_timeseries:ProcessBuilder):
    tsteps = list([input_timeseries.array_element(6*index) for index in range(0,6)])
    return array_concat(array_concat(input_timeseries.quantiles(probabilities=[0.1,0.5,0.9]),input_timeseries.sd()),tsteps)

def rf_classification(bbox, con=None, udf=udf_rf, year=2020):
    temp_ext = [str(year-1)+"-09-01", str(year+1)+"-04-30"]
    spat_ext = bbox

    ### Sentinel 2 data
    s2 = con.load_collection("TERRASCOPE_S2_TOC_V2",
                                temporal_extent=temp_ext,
                                spatial_extent=spat_ext,
                                bands=["B03","B04","B05","B06","B07","B08","B11","B12","SCL"])
    s2._pg.arguments['featureflags'] = temporal_partition_options
    s2 = s2.process("mask_scl_dilation", data=s2, scl_band_name="SCL").filter_bands(s2.metadata.band_names[:-1])

    ### Cropland mask
    cropland_mask = con.load_collection("TEST_LAYER",
                                    temporal_extent=temp_ext,
                                    spatial_extent=spat_ext,
                                    bands=["Map"]
                                    )
    cropland_mask._pg.arguments['featureflags'] = temporal_partition_options
    cropland_mask = cropland_mask.band("Map") != 40

    s2 = s2.mask(cropland_mask.resample_cube_spatial(s2).max_time())

    ### Base feature calculation
    idx_list = ["NDVI","NDMI","NDGI","ANIR","NDRE1","NDRE2","NDRE5"]
    s2_list = ["B06","B12"]

    indices = compute_indices(s2, idx_list, 250).filter_bands(s2_list+idx_list)
    idx_dekad = indices.aggregate_temporal_period(period="dekad", reducer="mean")
    idx_dekad = idx_dekad.apply_dimension(dimension="t", process="array_interpolate_linear").filter_temporal([str(year)+"-01-01", str(year)+"-12-31"])

    base_features = idx_dekad.rename_labels("bands",s2_list+idx_list)

    ### Advanced feature calculation
    features = base_features.apply_dimension(dimension='t',target_dimension='bands', process=computeStats)

    tstep_labels = [ "t"+ str(6*index) for index in range(0,6) ]
    features = features.rename_labels('bands',[band + "_" + stat for band in base_features.metadata.band_names for stat in ["p10","p50","p90","sd"] + tstep_labels ])

    clf_results = features.apply_dimension(code=udf_rf, runtime="Python", dimension="bands").rename_labels("bands",["pixel"]).apply(lambda x: x.linear_scale_range(0,250,0,250))
    return clf_results


def process_area(con=None, area=None, callback=None, tmp_ext=None, folder_path=None, frm="GTiff", minimum_area=0.5, parallel_jobs=1):
    """
    TODO: maak iets waarmee het asynchroon kan draaien?
    TODO: callback functie maken ipv hier in dit script draaien
    This function processes splits up the processing of large areas in several batch jobs.
    The function does not return anything but saves the batch jobs in a folder specified by folder_path
    and in the format specified by format

    :param con: a connection with one of the openeo backends
    :param area: a path to a geojson file
    :param nr_jobs: amount of batch jobs in which you want to split the area
    :param callback: a callback function with the operations that need to be run
    :param tmp_ext: the temporal extent for which to run the operation
    :param folder_path: folder_path in which to save the resulting batch jobs
    :param frm: the format in which to save the results
    """
    if frm=="netCDF":
        ext = ".nc"
    elif frm=="GTiff":
        ext = ".tif"
    else:
        raise NotImplementedError("This format is not yet implemented")
    geoms = gpd.read_file(area)
    # selective reading, requires geopandas>=0.7.0
    grid_tot = gpd.read_file("https://s3.eu-central-1.amazonaws.com/sh-batch-grids/tiling-grid-2.zip", mask=geoms)

    for index, geom in enumerate(geoms.geometry):
        
        intersection = grid_tot.geometry.intersection(geom)

        # filter on area
        grid_tot = grid_tot[intersection.area > minimum_area]

        bounds = geom.bounds
        grid = grid_tot.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].reset_index()
        print("The bounding box of the geometry you selected contains a total of " + str(
            len(grid)) + " grid tiles. Not all of these will actually overlap with the specific geometry shape you supplied")

        status_df = read_or_create_csv(grid, index)

        not_finished = len(status_df[status_df["status"] != "finished"])
        downloaded_results = []
        printed_errors = []

        while not_finished:
            def running_jobs():
                return status_df.loc[(status_df["status"] == "queued") | (status_df["status"] == "running")].index

            def update_statuses():
                default_usage = {
                    'cpu':{'value':0, 'unit':''},
                    'memory':{'value':0, 'unit':''}
                }
                for i in running_jobs():
                    job_id = status_df.loc[i, 'id']
                    job = con.job(job_id).describe_job()
                    usage = job.get('usage',default_usage)
                    status_df.loc[i, "status"] = job["status"]
                    status_df.loc[i, "cpu"] = f"{deep_get(usage,'cpu','value',default=0)} {deep_get(usage,'cpu','unit',default='')}"
                    status_df.loc[i, "memory"] = f"{deep_get(usage,'memory','value',default=0)} {deep_get(usage,'memory','unit',default='')}"
                    status_df.loc[i, "duration"] = deep_get(usage,'duration','value',default=0)
                    print(time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime()) + "\tCurrent status of job " + job_id
                          + " is : " + job["status"])

            def start_jobs(nb_jobs):
                pending_jobs = status_df[status_df["status"] == "pending"].index
                for i in range(min(len(pending_jobs), nb_jobs)):
                    tile = grid.geometry[pending_jobs[i]]
                    if geom.intersects(tile):
                        print("Starting job... current grid tile: " + str(pending_jobs[i]))
                        bbox = tile.bounds
                        lst = ('west', 'south', 'east', 'north')
                        bbox_dict = dict(zip(lst, bbox))

                        
                        cube = callback(bbox=bbox_dict, con=con)
                        s2_res = cube.save_result(format=frm)
                        job = s2_res.send_job(job_options={
                                "driver-memory": "2G",
                                "driver-memoryOverhead": "1G",
                                "driver-cores": "2",
                                "executor-memory": "3G",
                                "executor-memoryOverhead": "3G",
                                "executor-cores": "3",
                                # "queue": "lowlatency"
                                "max-executors": "90",
                                "queue": "archive", ## deze is nieuw!
                            },
                            title="5 countries using S2-only model, tile "+str(pending_jobs[i])+" with bbox "+str(bbox),
                            overviews="ALL",
                            colormap=col_palette
                        )
                        job.start_job()
                        status_df.loc[pending_jobs[i], "status"] = job.describe_job()["status"]
                        status_df.loc[pending_jobs[i], "id"] = job.describe_job()["id"]
                    else:
                        status_df.loc[pending_jobs[i], "status"] = "finished"

            def download_results(downloaded_results):
                finished_jobs = status_df[status_df["status"] == "finished"].index
                for i in finished_jobs:
                    if i not in downloaded_results:
                        downloaded_results += [i]
                        job_id = status_df.loc[i, 'id']
                        job = con.job(job_id)
                        print("Finished job: {}, Starting to download...".format(job_id))
                        results = job.get_results()
                        results.download_file(folder_path + grid.name[i] + ext)

            def print_errors(printed_errors):
                error_jobs = status_df[status_df["status"] == "error"].index
                for i in error_jobs:
                    if i not in printed_errors:
                        printed_errors += [i]
                        print("Encountered a failed job: " + status_df.loc[i, 'id'])

            update_statuses()
            start_jobs(parallel_jobs - len(running_jobs()))
            # download_results(downloaded_results)
            print_errors(printed_errors)

            not_finished = len(status_df[status_df["status"] != "finished"])
            average_duration = status_df.duration.mean()
            time_left = not_finished * average_duration / parallel_jobs
            print("Jobs not finished: " + str(not_finished))
            print("Average duration: " + str(average_duration) + " time left: " + str(time_left) + " seconds")

            status_df.to_csv(csv_path.format(index))

            time.sleep(45)


connection = openeo.connect("https://openeo-dev.vito.be")
connection.authenticate_oidc()
geom = 'UC3_resources/processing_area.geojson'
csv_path = "./data/uc3_job_status_dev{}.csv"
process_area(con=connection, area=geom, callback=rf_classification, folder_path="./data/large_areas/", minimum_area=0.7, parallel_jobs=2)
