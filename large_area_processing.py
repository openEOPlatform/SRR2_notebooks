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


def _callback(x: ProcessBuilder, bandnames) -> ProcessBuilder:
    ndvi_apr = x.array_element(bandnames.index("NDVI_apr"))
    ndvi_may = x.array_element(bandnames.index("NDVI_may"))
    ndvi_sep = x.array_element(bandnames.index("NDVI_sep"))
    ndvi_nov = x.array_element(bandnames.index("NDVI_nov"))

    nir_mar = x.array_element(bandnames.index("B08_mar"))
    nir_may = x.array_element(bandnames.index("B08_may"))
    swir_mar = x.array_element(bandnames.index("B11_mar"))
    swir_may = x.array_element(bandnames.index("B11_may"))
    
    corn_greenup = if_(ndvi_apr.lt(ndvi_may),1,0)
    corn_harvest = if_(ndvi_sep.gt(ndvi_nov),1,0)
    
    corn_sen_p1 = if_((nir_mar).gt(swir_may),1,0)
    corn_sen_p2 = if_((swir_mar).gt(nir_may),1,0)

    corn_total = corn_greenup + corn_harvest + corn_sen_p1 + corn_sen_p2
    corn_total_fin = corn_total.gt(2)
    return corn_total_fin


def process_callback(con=None):
    s2_masked = con.process("mask_scl_dilation", data=con, scl_band_name="SCL").filter_bands(["B04","B08", "B11"])
    ndvi_comp = compute_indices(s2_masked, ["NDVI"])
    ndvi_comp_byte = ndvi_comp.linear_scale_range(1,2000,0,254)
    agg_month = ndvi_comp_byte.aggregate_temporal_period(period="month", reducer="median")
    ndvi_month = agg_month.apply_dimension(dimension="t", process="array_interpolate_linear").filter_temporal([str(year)+"-01-01", str(year)+"-12-31"])
    all_bands = ndvi_month.apply_dimension(dimension='t', target_dimension='bands', process=lambda x: x*1)

    bandnames2 = [band + "_" + stat for stat in ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"] for band in all_bands.metadata.band_names]
    bandnames = [band + "_" + stat for band in all_bands.metadata.band_names for stat in ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]]
    all_bands = all_bands.rename_labels('bands', target=bandnames)

    corn = all_bands.reduce_dimension(dimension="bands", reducer=lambda x: _callback(x,bandnames))
    return corn


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

    for index, geom in enumerate(geoms.geometry):
        # selective reading, requires geopandas>=0.7.0
        grid_tot = gpd.read_file("https://s3.eu-central-1.amazonaws.com/sh-batch-grids/tiling-grid-2.zip", mask=geoms)
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

                        s2 = con.load_collection("TERRASCOPE_S2_TOC_V2",
                                                 spatial_extent=bbox_dict,
                                                 temporal_extent=tmp_ext,
                                                 bands=["B04", "B08", "B11", "SCL"])
                        s2._pg.arguments['featureflags'] = temporal_partition_options
                        cube = callback(s2)
                        s2_res = cube.save_result(format=frm)
                        job = s2_res.send_job(job_options={
                            "driver-memory": "2G",
                            "driver-memoryOverhead": "1G",
                            "driver-cores": "2",
                            "executor-memory": "2G",
                            "executor-memoryOverhead": "1G",
                            "executor-cores": "3",
                            # "queue": "lowlatency"
                            "max-executors": "100"
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
connection = openeo.connect("https://openeo.vito.be")
# connection.authenticate_oidc()
connection.authenticate_basic("XXX","XXX")
geom = 'UC3_resources/processing_area.geojson'
csv_path = "./data/uc3_job_status_{}.csv"
tmp_ext = [str(year-1)+"-11-01", str(year+1)+"-02-01"]
process_area(con=connection, area=geom, callback=process_callback, tmp_ext=tmp_ext, folder_path="./data/large_areas/")
