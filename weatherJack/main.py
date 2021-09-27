import numpy as np
import os
import pickle, gzip
import pandas as pd
import glob
import sys

class parameters(object):
    """Handle user provided parameters"""
    def __init__(self):
        # default values
        self.ID_WIND_EAST = "48758" # u component
        self.ID_WIND_NORTH = "48759" # v component
        self.ID_REL_HUMID = "48760" # relative humidity
        self.ID_CLOUD_COV = "48764" # fraction of cloud cover
        self.PRESSURE_LEVEL = "500" # in hPA, 500 hPa correspond around 5500 m which is in the middle of the troposphere, most relevant for most weather phenomenon

        # user parameters: area of interest (square)
        self.LA_S = -90.
        self.LO_W = -180.
        self.LA_N = 90.
        self.LO_E = 180.
        
        # TODO make user parameters safe (convert to python property for validity checking)
        assert -90 <= self.LA_S <= self.LA_N <= 90 and -180 <= self.LO_W <= self.LO_E <= 180
        
        self.MONTH_START = 1 # averaging from beginning of this month
        self.MONTH_END = 3 # averaging until beginning of this month (if same month, averaging over 1 year)

        assert self.MONTH_START != self.MONTH_END
        assert 1<=self.MONTH_START<=12 and 1<=self.MONTH_END<=12

        self.YEARS = np.array([2016,2017,2018]) # user data, years of the start month
        year_add = int(self.MONTH_END <= self.MONTH_START)
        assert self.YEARS.min()>=1999 and self.YEARS.max()+year_add<=2018 # available data from the ECMWF database (ends in August 2019)
        
        # algorithm tuning parameter
        self.CRIT_CC = 0.2 # critical cloud cover for score S1 calculation
        assert 0<self.CRIT_CC<=1
        self.CRIT_TD_FAC = 0.25 # prefactor critical traveling distance for score S2 calculation (sets importance of impact (small factor asks for high impact) in relation to costs), critical traveling distance needs to be larger than 0.5*minimum traveling distance
        


class weatherData(object):
    """Manages weather data locally"""
    
    def __init__(self, pairs=None, folder="weather_data", verbose=False):
        self.verbose = verbose
        try:
            os.makedirs(folder)
        except FileExistsError:
            pass # directory already exists
        self.__data_folder = folder
        
        if pairs!=None:
            # TODO: check if connection to server is working
            self.__ONLINE = True
            self.pairs = pairs
        else:
            self.__ONLINE = False
    
    
    def get_data(self, layer_id, level, year, month_start, month_end, clear_downloads=True, fix_data=True):
        """Returns a dictionary with key 'metadata' and 'data', fetches data if not locally available and server connection established."""
        file_id = f"{self.__data_folder}/{layer_id}_{level}_{year}_{str(month_start).zfill(2)}_{str(month_end).zfill(2)}"
        file_exists = os.path.isfile(file_id)
        if file_exists:
            if self.verbose:
                sys.stdout.writelines("Read data from cache...\n")
            with gzip.open(file_id, "rb") as f:
                return pickle.load(f)
        
        elif not file_exists and self.__ONLINE:
            if self.verbose:
                sys.stdout.writelines("Data not found, fetch from server...\n")
            year_add = int(month_end <= month_start)
            query_json = {
                "layers": [{
                    "type": "raster",
                    "id": str(layer_id),
                    "temporal": {
                        "intervals": [{
                            "start": f"{year}-{str(month_start).zfill(2)}-01T00:00:00Z",
                            "end": f"{year+year_add}-{str(month_end).zfill(2)}-01T00:00:00Z"
                        }]
                    },
                    "aggregation": "Mean",
                    "dimensions": [{
                        "name": "level",
                        "value": level
                    }]
                }],
                "spatial": {
                    "type": "square",
                    "coordinates" : ["-90.0", "-180.0", "90.0", "180.0"]
                },
                "temporal" : {
                    "intervals" : [{
                        "start": f"{year}-{str(month_start).zfill(2)}-01T00:00:00Z",
                        "end": f"{year+year_add}-{str(month_end).zfill(2)}-01T00:00:00Z"
                    }]
                }
            }
            query = self.pairs.query(query_json)
            query.submit()
            query.poll_till_finished()
            query.download()
            query.create_layers()
            
            query_metadata = pd.DataFrame(query.metadata).transpose()
            id_string = query_metadata[(query_metadata['datalayerId'] == layer_id)].index[0]
            
            data = {}
            data["metadata"] = query.metadata[id_string]
            data["data"] = query.data[id_string]
            if fix_data:
                data["data"][np.isnan(data["data"])] = 0.
            
            if clear_downloads:
                files = glob.glob("downloads/*")
                for f in files:
                    os.remove(f)
            
            with gzip.open(file_id, "wb") as f:
                pickle.dump(data, f)
            
            return data
        
        else:
            sys.stderr.writelines("No connection to database established and data not available locally. Exit.\n")
            sys.exit(1)
    
    
    def year_average_data(self, layer_id, level, years, month_start, month_end):
        """Average data over years given in array years"""
        data = self.get_data(layer_id, level, years[0], month_start, month_end)
        avg_data = data["data"]
        if self.verbose:
            print("Year averaging...")
            print(f"1/{len(years)}")
        for idx, y in enumerate(years[1:]):
            avg_data += self.get_data(layer_id, level, y, month_start, month_end)["data"]
            if self.verbose:
                print(f"{idx+2}/{len(years)}")
        avg_data /= len(years)
        return avg_data



class analyzer(object):
    """Main analysis logic of weather jacking"""
    def __init__(self, weather_data, user_parameters):
        self.N_STEPS = 300
        self.R_EARTH = 6371 * 1e3 # mean earth radius in meter
        self.VEL_TYP = 10. # typical intermediate wind velocity in meter/second (used as unit to travel one pixel per step in longitude on the equator)
        self.C1 = np.pi/180. # multiplication transforms degree to rad

        self.reset(weather_data, user_parameters)
        
    def reset(self, weather_data, user_parameters):
        self.wd = weather_data
        self.par = user_parameters
        # get metadata information
        metadata = self.wd.get_data(self.par.ID_WIND_EAST, self.par.PRESSURE_LEVEL, self.par.YEARS[0], self.par.MONTH_START, self.par.MONTH_END)["metadata"]
        self.pxl_size = metadata["details"]["pixelDimensions"]["pixelSizeDegreeLongitude"] # implies same size for longitude and latitude for all layers
        self.n_lo_data = metadata["details"]["pixelDimensions"]["numberPixelsLongitude"]
        self.n_la_data = metadata["details"]["pixelDimensions"]["numberPixelsLatitude"]
        self.lo_w_data, self.lo_e_data, self.la_s_data, self.la_n_data = [
            metadata["details"]["boundingBox"][k]
            for k in ["minLongitude", "maxLongitude", "minLatitude", "maxLatitude"]
        ]
        self.dt = np.pi * self.R_EARTH/self.VEL_TYP * self.pxl_size/180. # time step
        
        # get average weather data
        self.u_data = self.wd.year_average_data(self.par.ID_WIND_EAST, self.par.PRESSURE_LEVEL, self.par.YEARS, self.par.MONTH_START, self.par.MONTH_END) # longitude wind
        self.v_data = self.wd.year_average_data(self.par.ID_WIND_NORTH, self.par.PRESSURE_LEVEL, self.par.YEARS, self.par.MONTH_START, self.par.MONTH_END) # latitude wind
        self.h_data = self.wd.year_average_data(self.par.ID_REL_HUMID, self.par.PRESSURE_LEVEL, self.par.YEARS, self.par.MONTH_START, self.par.MONTH_END)/100. # relative humidity
        self.c_data = self.wd.year_average_data(self.par.ID_CLOUD_COV, self.par.PRESSURE_LEVEL, self.par.YEARS, self.par.MONTH_START, self.par.MONTH_END) # fraction of cloud cover
    
    def __step(self, lo, la, u, v):
        """Internal method to perform a step for all grid points and update of lo, la"""
        d_lo = u * self.dt / (self.C1*np.cos(la*self.C1)*self.R_EARTH) # change in longitude
        d_la = v * self.dt / (self.C1*self.R_EARTH) # change in latitude
        lo = (lo+180.+d_lo)%360. - 180.
        la = np.arcsin(np.sin((la+d_la)*self.C1))/self.C1
        return lo, la
    
    def __u_v_update(self, lo, la, u, v):
        """Internal method to update the wind velocity components u (longitude, towards east) and v (latitude, towards north)."""
        # naive implementation from tutorial https://pairs.res.ibm.com/tutorial/tutorials/api/raster_data.html
        #i_lo = np.round((lo-lo_w_data)/(lo_e_data-lo_w_data) * n_lo_data - 0.5).astype(np.int64)
        #i_la = np.round((la-la_n_data)/(la_s_data-la_n_data) * n_la_data - 0.5).astype(np.int64) # NOTE: 0 index is most north (missing in tutorial)
        # corrected implementation (fixed rounding issue at the boundary)
        i_lo = np.round((lo-self.lo_w_data)/(self.lo_e_data-self.lo_w_data) * (self.n_lo_data-0.5) - 0.5).astype(np.int64)
        i_la = np.round((la-self.la_n_data)/(self.la_s_data-self.la_n_data) * (self.n_la_data-0.5) - 0.5).astype(np.int64)
        u = self.u_data[i_la,i_lo]
        v = self.v_data[i_la,i_lo]
        return u, v
    
    def great_circle_distance_rad(self, lo1, la1, lo2, la2):
        """Arguments given in radians"""
        return self.R_EARTH * np.arccos(np.sin(la1)*np.sin(la2)+np.cos(la1)*np.cos(la2)*np.cos(lo2-lo1))

    def great_circle_distance(self, lo1, la1, lo2, la2):
        """Arguments given in degrees"""
        return self.great_circle_distance_rad(self.C1*lo1, self.C1*la1, self.C1*lo2, self.C1*la2)
    
    def compute_user_grids(self):
        """Internal method to calculate the user grids (humidity, cloud cover, traveling distances for each point)."""
        # TODO: write analogue to compute only points instead of grid/area
        # indices of user defined area within full area
        i0_lo = round((self.par.LO_W-self.lo_w_data)/(self.lo_e_data-self.lo_w_data) * (self.n_lo_data-0.5) - 0.5)
        i1_lo = round((self.par.LO_E-self.lo_w_data)/(self.lo_e_data-self.lo_w_data) * (self.n_lo_data-0.5) - 0.5)
        n_lo = i1_lo-i0_lo+1
        i0_la = round((self.par.LA_N-self.la_n_data)/(self.la_s_data-self.la_n_data) * (self.n_la_data-0.5) - 0.5)
        i1_la = round((self.par.LA_S-self.la_n_data)/(self.la_s_data-self.la_n_data) * (self.n_la_data-0.5) - 0.5)
        n_la = i1_la-i0_la+1
        
        # grid windows of global grid
        humidity_grid = self.h_data[i0_la:i1_la+1, i0_lo:i1_lo+1]
        cloud_grid = self.c_data[i0_la:i1_la+1, i0_lo:i1_lo+1]
        
        # traveling distance grid
        los = np.linspace(self.par.LO_W, self.par.LO_E, n_lo)
        las = np.linspace(self.par.LA_N, self.par.LA_S, n_la)
        la_grid, lo_grid = np.meshgrid(las, los, indexing="ij")
        la_grid_buffer, lo_grid_buffer = np.copy(la_grid), np.copy(lo_grid)
        u_grid, v_grid = np.empty_like(la_grid), np.empty_like(la_grid)
        u_grid, v_grid = self.__u_v_update(lo_grid, la_grid, u_grid, v_grid)
        dist_grid = np.zeros_like(la_grid)
        # ready for the vectorized sky walk(s)
        for i in range(self.N_STEPS):
            lo_grid, la_grid = self.__step(lo_grid, la_grid, u_grid, v_grid)
            #dist_grid += self.great_circle_distance(lo_grid_buffer, la_grid_buffer, lo_grid, la_grid)
            #la_grid_buffer, lo_grid_buffer = np.copy(la_grid), np.copy(lo_grid) # alternatively, the distance could be calculated just once from the end-to-end points (problem: multiple traveling around earth not tracked)
            u_grid, v_grid = self.__u_v_update(lo_grid, la_grid, u_grid, v_grid)
        dist_grid = self.great_circle_distance(lo_grid_buffer, la_grid_buffer, lo_grid, la_grid) # instant traveling distances of a massless particle spawned at the different grid points
        # cap too small values of the traveling distances
        td_min = dist_grid.min()
        td_min = 1. if td_min < 1. else td_min
        dist_grid[dist_grid < td_min] = td_min
        
        return humidity_grid, cloud_grid, dist_grid
    
    def compute_score1(self, relative_humidity, cloud_cover, traveling_distance, td_min=None, td_max=None):
        """Computes the S1 score (between 0 and 1) for ADDING water / clouds
        linear combination of relative humidity (rh), fraction of cloud cover (cc), and traveling distance (td)
        S1 = a1 rh - a2 cc + a3 td
        with a1,a2,a3 > 0
        The higher the score, the more interesting is the insertion of clouds.
        High rel. humidity makes cloud formation easy, already high cloud coverage 
        makes cloud formation less interesting, large traveling distances indicate 
        a high impact on the environment.
        Preprocessing: all points with td<td_min are neglected and get score 0
        Introduce critical cloud cover cc' above which there is no benefit of adding water 
        no matter how cheap (high humidity) it is:
        a1 1.0 - a2 cc' = 0 -> a2 = a1/cc'
        Fix scores to the range [0,1] gives conditions for a1 and a3 (cc' free parameter)
        """
        td_min = traveling_distance.min() if td_min==None else td_min
        td_max = traveling_distance.max() if td_max==None else td_max
        
        # coefficients from analytic solution
        A1 = self.par.CRIT_CC/(td_max/td_min + self.par.CRIT_CC)
        A2 = A1/self.par.CRIT_CC
        A3 = (1.-A1)/td_max
        
        return A1*relative_humidity - A2*cloud_cover + A3*traveling_distance
    
    def compute_score2(self, relative_humidity, cloud_cover, traveling_distance, td_min=None, td_max=None):
        """
        Computes the S2 score for REMOVING water / clouds
        Modified linear combination as for S1:
        S2' = b1 rh + b2 cc+ b3 td
        with b1,b2,b3 > 0
        The higher the score, the more interesting is the removing of clouds.
        In case the relative humidity or the cloud coverage is high, it is easy 
        to remove clouds / water locally. Therefore assume b1=b2 (assume it is as easy to 
        get water from humidity as from clouds). Introduce td_crit = b1/b3,
        with td_crit > 2 td_min
        which is a characteristic traveling distance that puts the importance of 
        the impact (td) in relation to the cost (low rh and cc have high cost)
        In case the cost is optimal (rh=cc=1) but there is no impact (td->0)
        the score is s2 = 1/(1+td_max/(2 b - td_min)). A good estimate for td_crit
        is therefore 
        2 td_crit = td_max/2 (s2 ~ 1/3) or 2 td_crit = td_max (s2 ~ 1/2)
        Also, s2 is restricted to [0,1], modified s2 equation (used in the code) is
        S2 = b1 (rh + cc + td/td_crit) - s2_offset
        with s2_offset = b1 td_min/td_crit
        Requiring max(S2) = 1 gives
        b1 = 1 / (2 + (td_max-td_min)/td_crit)
        """
        td_min = traveling_distance.min() if td_min==None else td_min
        td_max = traveling_distance.max() if td_max==None else td_max
        td_crit = self.par.CRIT_TD_FAC*td_max
        assert td_crit > 2*td_min
        
        # coefficients from analytic solution
        B1 = 1./(2.+(td_max-td_min)/td_crit)
        B2 = B1
        B3 = B1/td_crit
        S2_offset = B1*td_min/td_crit
        
        return B1*relative_humidity + B2*cloud_cover + B3*traveling_distance - S2_offset
    
    def find_maxs(self, grid_original, n_max=5, excl_radius=30):
        grid = np.copy(grid_original)
        #grid = grid_original # for DEBUGGING
        max_is, max_js = np.zeros(5,dtype=np.int64), np.zeros(5,dtype=np.int64)
        grid_min = grid.min()
        ni_grid, nj_grid = grid.shape
        for i in range(n_max):
            max_is[i], max_js[i] = np.unravel_index(grid.argmax(), grid.shape)
            i0 = max(0, max_is[i]-excl_radius)
            i1 = min(ni_grid-1, max_is[i]+excl_radius)
            j0 = max(0, max_js[i]-excl_radius)
            j1 = min(nj_grid-1, max_js[i]+excl_radius)
            grid[i0:i1, j0:j1] = grid_min
        
        # from array indices to latitude and longitude
        i0_lo = round((self.par.LO_W-self.lo_w_data)/(self.lo_e_data-self.lo_w_data) * (self.n_lo_data-0.5) - 0.5)
        i1_lo = round((self.par.LO_E-self.lo_w_data)/(self.lo_e_data-self.lo_w_data) * (self.n_lo_data-0.5) - 0.5)
        n_lo = i1_lo-i0_lo+1
        i0_la = round((self.par.LA_N-self.la_n_data)/(self.la_s_data-self.la_n_data) * (self.n_la_data-0.5) - 0.5)
        i1_la = round((self.par.LA_S-self.la_n_data)/(self.la_s_data-self.la_n_data) * (self.n_la_data-0.5) - 0.5)
        n_la = i1_la-i0_la+1
        
        los = np.linspace(self.par.LO_W, self.par.LO_E, n_lo)
        las = np.linspace(self.par.LA_N, self.par.LA_S, n_la)
        #la_grid, lo_grid = np.meshgrid(las, los, indexing="ij")
        
        la_maxs = las[max_is]
        lo_maxs = los[max_js]
        
        return la_maxs, lo_maxs
        
        
            
            
        



