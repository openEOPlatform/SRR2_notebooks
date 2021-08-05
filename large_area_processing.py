import openeo
import geopandas as gpd
import pandas as pd
import time
from helper import compute_indices
from openeo.processes import ProcessBuilder, if_

temporal_partition_options = {
        "indexreduction": 0,
        "temporalresolution": "None",
        "tilesize": 256
    }
default_partition_options = {
    "tilesize": 256
}


def read_or_create_csv(grid, index):
    try:
        status_df = pd.read_csv(csv_path.format(index), index_col=0)
    except FileNotFoundError:
        status_df = pd.DataFrame(columns=["name", "status", "id", "cpu", "memory"])

        for i in range(len(grid)):
            status_df = status_df.append(
                {"name": grid.name[i], "status": "pending", "id": None, "cpu": None, "memory": None}, ignore_index=True)

        status_df.to_csv(csv_path.format(index))

    return status_df



def sentinel2_stratification(bbox_dict, con=None):
    s2 = con.load_collection("TERRASCOPE_S2_TOC_V2",
                                                 spatial_extent=bbox_dict,
                                                 temporal_extent=tmp_ext,
                                                 bands=["B04", "B08", "B11", "SCL"])
    s2._pg.arguments['featureflags'] = temporal_partition_options
        
    s2_masked = s2.process("mask_scl_dilation", data=s2, scl_band_name="SCL").filter_bands(["B04","B08", "B11"])
    ndvi_comp = compute_indices(s2_masked, ["NDVI"])
    ndvi_comp_byte = ndvi_comp.linear_scale_range(1,2000,0,250)
    
    agg_month = ndvi_comp_byte.aggregate_temporal_period(period="month", reducer="mean")
    ndvi_month = agg_month.apply_dimension(dimension="t", process="array_interpolate_linear").filter_temporal([str(year)+"-01-01", str(year)+"-12-31"])
    
    all_bands = ndvi_month.apply_dimension(dimension='t', target_dimension='bands', process=lambda x: x*1)
    bandnames2 = [band + "_" + stat for stat in ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"] for band in all_bands.metadata.band_names]
    bandnames = [band + "_" + stat for band in all_bands.metadata.band_names for stat in ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]]
    all_bands = all_bands.rename_labels('bands', target=bandnames)

    ndvi_apr = all_bands.band("NDVI_apr")
    ndvi_may = all_bands.band("NDVI_may")
    ndvi_jun = all_bands.band("NDVI_jun")
    ndvi_jul = all_bands.band("NDVI_jul")
    ndvi_aug = all_bands.band("NDVI_aug")
    ndvi_sep = all_bands.band("NDVI_sep")
    ndvi_oct = all_bands.band("NDVI_oct")
    ndvi_nov = all_bands.band("NDVI_nov")

    nir_mar = all_bands.band("B08_mar")
    nir_may = all_bands.band("B08_may")
    nir_jun = all_bands.band("B08_jun")
    nir_oct = all_bands.band("B08_oct")
    swir_mar = all_bands.band("B11_mar")
    swir_may = all_bands.band("B11_may")
    swir_oct = all_bands.band("B11_oct")

    ## Rule for corn is not in line with experiment for 2019, use ndvi_may < ndvi_jun
    corn = (((ndvi_may < ndvi_jun) + (ndvi_sep > ndvi_nov) + (nir_mar > swir_may) + (swir_mar > nir_may)) == 4)*1
    barley = (((ndvi_apr < ndvi_may) + (ndvi_jul < ndvi_jun)) == 2)*1 ## barley has very narrow and early NDVI 
    sugarbeet = (((ndvi_may < 0.6*ndvi_jun) + ((ndvi_jun+ndvi_jul+ndvi_aug+ndvi_sep)/4 > 0.7))==2)*1 #4 month period of high NDVI
    potato = ((((ndvi_jun/ndvi_may) > 2) + (ndvi_sep < ndvi_jul) + (ndvi_nov > (ndvi_sep + ndvi_oct)/2) + ((swir_may / nir_may) > 0.8) + ((nir_jun / nir_may) > 1.5)) == 5)*1
    soy = ((((ndvi_may / ndvi_apr) < 1.2) + ((ndvi_may / ndvi_apr) > 0.8) + (ndvi_sep < ndvi_aug) + ((nir_oct / swir_oct) < 1.1)) == 4)*1

    total = 1*corn + 2*barley + 4*sugarbeet + 8*potato + 16*soy #allow multiple crops to be detected...
    
    return total


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
                for i in running_jobs():
                    job_id = status_df.loc[i, 'id']
                    job = con.job(job_id).describe_job()
                    status_df.loc[i, "status"] = job["status"]
                    status_df.loc[i, "cpu"] = f"{job['usage']['cpu']['value']} {job['usage']['cpu']['unit']}"
                    status_df.loc[i, "memory"] = f"{job['usage']['memory']['value']} {job['usage']['memory']['unit']}"
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

                        
                        cube = callback(bbox_dict, con)
                        s2_res = cube.save_result(format=frm)
                        job = s2_res.send_job(job_options={
                            "driver-memory": "2G",
                            "driver-memoryOverhead": "1G",
                            "driver-cores": "2",
                            "executor-memory": "2G",
                            "executor-memoryOverhead": "1G",
                            "executor-cores": "3",
                            # "queue": "lowlatency"
                            "max-executors": "70"
                        })
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
            download_results(downloaded_results)
            print_errors(printed_errors)

            not_finished = len(status_df[status_df["status"] != "finished"])
            print("Jobs not finished: " + str(not_finished))

            status_df.to_csv(csv_path.format(index))

            time.sleep(45)


year = 2020
connection = openeo.connect("https://openeo-dev.vito.be")
# connection.authenticate_oidc()
connection.authenticate_basic("driesj","driesj123")
geom = 'UC3_resources/processing_area.geojson'
csv_path = "./data/uc3_job_status_{}.csv"
tmp_ext = [str(year-1)+"-11-01", str(year+1)+"-02-01"]
process_area(con=connection, area=geom, callback=sentinel2_stratification, tmp_ext=tmp_ext, folder_path="./data/large_areas/", minimum_area=0.7, parallel_jobs=3)
