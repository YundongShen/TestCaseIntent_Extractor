from locust import HttpUser, between, task, tag


class ESGFUser(HttpUser):
    host = "http://esgf3.dkrz.de"
    wait_time = between(5, 15)

    @task
    def catalog(self):
        query = "/thredds/catalog/catalog.html"

        with self.client.get(query, catch_response=True, name="catalog") as response:
            if "Earth System Root catalog" not in response.text:
                response.failure("Catalog response not as expected")

    @tag("cmip6")
    @task
    def cmip6head(self):
        query = "/thredds/fileServer/cmip6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/tas/gn/v20190710/tas_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001-186912.nc"
        with self.client.head(query, catch_response=True, name="cmip6head") as response:
            if "application/x-netcdf" != response.headers.get('Content-Type'):
                response.failure("cmip6 response not as expected")

    @tag("cmip6")
    @task
    def cmip6get1(self):
        query = "/thredds/fileServer/cmip6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/tas/gn/v20190710/tas_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_185001-186912.nc"
        with self.client.get(query, catch_response=True, name="cmip6get1") as response:
            if "application/x-netcdf" != response.headers.get('Content-Type'):
                response.failure("cmip6 response not as expected")

    @tag("cmip6")
    @task
    def cmip6get2(self):
        query = "/thredds/fileServer/cmip6/ScenarioMIP/DWD/MPI-ESM1-2-HR/ssp126/r2i1p1f1/Amon/hfss/gn/v20190710/hfss_Amon_MPI-ESM1-2-HR_ssp126_r2i1p1f1_gn_209501-209912.nc"
        with self.client.get(query, catch_response=True, name="cmip6get2") as response:
            if "application/x-netcdf" != response.headers.get('Content-Type'):
                response.failure("cmip6 response not as expected")

    @tag("cmip5")
    @task
    def cmip5get1(self):
        query = "/thredds/fileServer/cmip5/cmip5/output1/MPI-M/MPI-ESM-LR/decadal1990/fx/land/fx/r0i0p0/v20111122/sftgif/sftgif_fx_MPI-ESM-LR_decadal1990_r0i0p0.nc"
        with self.client.get(query, catch_response=True, name="cmip5get1") as response:
            if not response.ok:
            #if "application/x-netcdf" != response.headers.get('Content-Type'):
                response.failure(f"cmip5 response not as expected {response.headers}")

    @tag("cmip5")
    @task
    def cmip5get2(self):
        query = "/thredds/fileServer/cmip5/cmip5/output1/MPI-M/MPI-ESM-LR/decadal1990/mon/ocean/Omon/r3i1p1/v20120529/tauvo/tauvo_Omon_MPI-ESM-LR_decadal1990_r3i1p1_199101-200012.nc"
        with self.client.get(query, catch_response=True, name="cmip5get2") as response:
            if not response.ok:
            #if "application/x-netcdf" != response.headers.get('Content-Type'):
                response.failure(f"cmip5 response not as expected {response.headers}")




