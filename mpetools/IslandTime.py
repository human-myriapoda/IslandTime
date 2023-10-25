"""
TODO: ADD DESCRIPTION
TODO: Island with the same name
TODO: Read `.kmz.` file from Duvat & Magnan (2019)
TODO: Select appropriate climate indices -> confounders

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import warnings
warnings.filterwarnings("ignore")
import os
import pickle
import pandas as pd
import re
import urllib.request
import numpy as np
import datetime
import ee
import wbgapi as wb
import requests
import json
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
import xarray as xr
import cdsapi
import osmnx as ox
import shapely
import pyproj
from scipy import interpolate
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from coastsatmaster.coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects
import geopy.distance
import geopandas as gpd
from wikidataintegrator import wdi_core
import geojson
from osgeo import gdal
import cartopy.crs as ccrs

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()
    ee.Initialize()

# Other commands
plt.ion()

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class IslandTimeBase:
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        """
        Initialise IslandTimeBase with island and country information.

        Args:
            island (str): The name of the island.
            country (str): The name of the country.
            verbose_init (bool): Whether to print verbose messages when retrieving island information.
            overwrite (bool): Whether to overwrite existing data.
        """
        self.island = island
        self.country = country
        self.verbose_init = verbose_init
        self.island_info_path = os.path.join(os.getcwd(), 'data', 'info_islands')
        self.overwrite = overwrite 

    def assign_metadata(self):
        """
        Assign metadata to the time series and information.

        This method assigns metadata to the island information based on the values of attributes like `acronym`, 
        `description`, and `source`.
        """        
        if self.acronym == 'ECU':
            self.island_info['characteristics_{}'.format(self.acronym)]['description'] = self.description
            self.island_info['characteristics_{}'.format(self.acronym)]['source'] = self.source
            
        else:
            self.island_info['timeseries_{}'.format(self.acronym)]['description'] = self.description
            self.island_info['timeseries_{}'.format(self.acronym)]['description_timeseries'] = self.description_timeseries
            self.island_info['timeseries_{}'.format(self.acronym)]['source'] = self.source
    
    def other_info_in_island_info_path(self):
        """
        Retrieve data from other islands in the same country.

        This method checks if there is information about another island from the same country in a specified path
        (`self.island_info_path`). If such information is found, it returns the time series data for that island.

        Returns:
            dict or None: Dictionary with timeseries data if found, None otherwise.
        """
        if np.shape(np.argwhere(np.array([self.country in listt for listt in os.listdir(self.island_info_path)])))[0] > 2:

            # Retrieve dictionary for another island of that country
            array_listdir = np.array(os.listdir(self.island_info_path))
            array_listdir = np.delete(array_listdir, np.argwhere(array_listdir == 'info_{}_{}.data'.format(self.island, self.country)))
            idx_listdir = np.argwhere(np.array([self.country in listt for listt in array_listdir]))
            
            for idx_file in idx_listdir:
                with open(os.path.join(self.island_info_path, str(array_listdir[idx_file][0])), 'rb') as fw:
                    island_info_other_island = pd.read_pickle(fw)

                # Fill information with other island
                if 'timeseries_{}'.format(self.acronym) in list(island_info_other_island.keys()):
                    print('~ Time series retrieved from another island (same country). ~')
                    return island_info_other_island['timeseries_{}'.format(self.acronym)]['timeseries']
        
        return None

    def main(self):
        """
        Main execution logic of IslandTimeBase.

        This method performs the main tasks of retrieving island information, handling metadata, and saving the data.
        """
        # Retrieve the dictionary with currently available information about the island
        self.island_info = retrieve_island_info(self.island, self.country, verbose=self.verbose_init)

        print('\n-------------------------------------------------------------------')
        print('Retrieving {}'.format(self.name_description))
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        if self.acronym == 'ECU':
            if not 'characteristics_{}'.format(self.acronym) in self.island_info.keys():
                # Create key/dict for data
                self.island_info['characteristics_{}'.format(self.acronym)] = {}

                # Assign metadata
                self.assign_metadata()

                # Run all functions
                self.add_info()

            else:
                if self.overwrite:
                    self.add_info() # run all functions
            
                else:
                    print('~ Information already available. Returning data. ~')

        else:             
            # If data have NOT already been generated
            if not 'timeseries_{}'.format(self.acronym) in self.island_info.keys():

                # Create key/dict for data
                self.island_info['timeseries_{}'.format(self.acronym)] = {}

                # Assign metadata
                self.assign_metadata()

                # Run all functions
                self.get_timeseries()

            elif not 'timeseries' in self.island_info['timeseries_{}'.format(self.acronym)].keys():
                # Run all functions
                self.get_timeseries()
            
            # If data have already been generated
            else:
                if self.overwrite:
                    self.get_timeseries() # run all functions

                else:
                    print('~ Information already available. Returning data. ~')

        # Save dictionary
        with open(os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country)), 'wb') as f:
            pickle.dump(self.island_info, f)
        
# --------------------------------------------------------------------------------------------------------------------------------------------------------
class PreTimeSeries:
    def __init__(self, island, country, alt_name=None, to_do=None, \
                 dict_geometry={'var_init': 0.001, 'var_change': 0.0005, 'thresold_ndvi': 0.2, 'var_limit': 0.065}, \
                 list_sat_user=None, date_range_user=None, cloud_threshold=10, image_type='SR', method='new', \
                 relevant_properties_wikidata=['P361', 'P131', 'P206', 'P2044'], \
                 verbose=True, wikidata_id=None, overwrite=False, atoll=None, polygon=None):
        """
        Initialise a PreTimeSeries instance.

        Parameters:
        - island (str): The name of the island.
        - country (str): The name of the country the island is in.
        - alt_name (str, optional): An alternative name for the island.
        - to_do (dict, optional): A dictionary specifying which tasks to perform (e.g., {'coordinates': True, 'polygon': True}).
        - dict_geometry (dict, optional): A dictionary of geometry-related parameters.
        - list_sat_user (list, optional): A list of specific satellite names to consider.
        - date_range_user (list, optional): A custom date range for satellite image filtering.
        - cloud_threshold (int, optional): The cloud cover threshold for satellite image filtering.
        - image_type (str, optional): The type of satellite image to use (e.g., 'SR' for surface reflectance).
        - method (str, optional): The method for calculating geometry ('new' or 'old').
        - relevant_properties_wikidata (list, optional): Relevant Wikidata properties to fetch.
        - verbose (bool, optional): Whether to display verbose output.
        - wikidata_id (str, optional): The Wikidata ID for the island.
        - overwrite (bool, optional): Whether to overwrite existing data.
        """
        self.island = island
        self.country = country
        self.alt_name = alt_name
        self.to_do = to_do
        self.dict_geometry = dict_geometry
        self.list_sat_user = list_sat_user
        self.date_range_user = date_range_user
        self.cloud_threshold = cloud_threshold
        self.image_type = image_type
        self.method = method
        self.relevant_properties_wikidata = relevant_properties_wikidata
        self.verbose = verbose
        self.wikidata_id = wikidata_id
        self.overwrite = overwrite
        self.atoll = atoll
        self.polygon = polygon
        self.dict_satellite_path = os.path.join(os.getcwd(), 'data', 'info_satellite')
        self.island_info_path = os.path.join(os.getcwd(), 'data', 'info_islands')
        self.duvat_magnan_2019_path = os.path.join(os.getcwd(), 'data', 'duvat_magnan_2019')

    def retrieve_coordinates_wikipedia(self):
        """
        Retrieve island coordinates from Wikipedia.

        Returns:
        - float: Latitude of the island.
        - float: Longitude of the island.
        """
        # To fit Wikipedia's format
        place = self.island.replace(" ", "_")

        # Try this syntax: only the name of the island
        try:
            web_url = urllib.request.urlopen('https://en.wikipedia.org/wiki/{}'.format(place))

        # Try this syntax: the name of the island + _(island)
        except:
            try:
                web_url = urllib.request.urlopen('https://en.wikipedia.org/wiki/{}_(island)'.format(place))

            # Wikipedia page doesn't exist
            except:
                lat = np.nan
                lon = np.nan

                return lat, lon

        # Read data from website
        data_url = str(web_url.read())

        # Regex patterns for latitude and longitude
        pattern_lat = '"wgCoordinates":{"lat":(.*),"lon":'
        pattern_lon = '"lon":(.*)},"wg' #EditSubmit

        # Find the coordinates from the patterns
        try:
            lat = re.search(pattern_lat, data_url).group(1)
            lon = re.search(pattern_lon, data_url).group(1)
            lat = lat.replace('\\n', '')
            lon = lon.replace('\\n', '')

        # No patterns found
        except: 
            lat = np.nan
            lon = np.nan

        return lat, lon

    def retrieve_coordinates_geokeo(self):
        """
        Retrieve island coordinates from GeoKeo.

        Returns:
        - float: Latitude of the island.
        - float: Longitude of the island.
        """
        # To make sure we find the island and not another place with the same name
        place = "{}, {}".format(self.island, self.country)
        
        try:
            # Request url (from GeoKeo website)
            url_gk = 'https://geokeo.com/geocode/v1/search.php?q={}&api=YOUR_API_KEY'.format(place)
            resp = requests.get(url=url_gk)
            data_url = resp.json()

            # Retrieve the coordinates (from GeoKeo website)
            if 'status' in data_url:
                if data_url['status'] == 'ok':
                    lat = data_url['results'][0]['geometry']['location']['lat']
                    lon = data_url['results'][0]['geometry']['location']['lng']

                else:
                    lat, lon = np.nan, np.nan

            else:
                lat, lon = np.nan, np.nan

        except:
            lat, lon = np.nan, np.nan

        return lat, lon

    def coordinates_consensus(self):
        """
        Combine coordinates from Wikipedia and GeoKeo, resolving discrepancies.

        Returns:
        - float: Latitude of the island.
        - float: Longitude of the island.
        """
        # Extract the coordinates from Wikipedia
        lat_w, lon_w = self.retrieve_coordinates_wikipedia()
        if self.verbose: 
            print('Coordinates from Wikipedia (lat/lon):', lat_w, lon_w)

        # Extract the coordinates from GeoKeo
        lat_gk, lon_gk = self.retrieve_coordinates_geokeo()
        if self.verbose: 
            print('Coordinates from GeoKeo (lat/lon):', lat_gk, lon_gk)

        # Compare the values and if they are too different -> visual inspection
        if not np.nan in (lat_w, lon_w, lat_gk, lon_gk):
            lat_close = np.allclose(np.array(lat_w, dtype=float), np.array(lat_gk, dtype=float), atol=1e-1)
            lon_close = np.allclose(np.array(lon_w, dtype=float), np.array(lon_gk, dtype=float), atol=1e-1)

            if not (lat_close and lon_close):
                ver = input('Is this {}, {} [y/n/wiki]? {}, {} - Please verify those coordinates on Google Maps: '.format(self.island, self.country, lat_gk, lon_gk))

                if ver == 'n':
                    lat, lon = input('Please enter the latitude/longitude: ').replace(', ', ' ').split(' ')

                elif ver == 'wiki':
                    lat, lon = lat_w, lon_w        

                else:
                    lat, lon = lat_gk, lon_gk

            else:
                lat, lon = lat_gk, lon_gk

        else:
            ver = input('Is this {}, {} [y/n/wiki]? {}, {} - Please verify those coordinates on Google Maps: '.format(self.island, self.country, lat_gk, lon_gk))

            if ver == 'n':
                lat, lon = input('Please enter the latitude/longitude: ').replace(', ', ' ').split(' ')

            elif ver == 'wiki':
                lat, lon = lat_w, lon_w

            else:
                lat, lon = lat_gk, lon_gk

        return lat, lon

    def retrieve_coordinates(self):
        """
        Retrieve island coordinates using the specified method (new or old).

        Updates the island_info dictionary with latitude and longitude.
        """        
        # New method (using OpenStreetMap)
        if self.method == 'new':
            try:
                area = ox.geocode_to_gdf(self.island + ', ' + self.country)
                lat, lon = area.lat[0], area.lon[0]

            except:
                # We call the function `coordinates_consensus` to retrieve lat, lon (old method)
                print('Island not available in OpenStreetMap. Will use other methods.')
                lat, lon = self.coordinates_consensus()
        
        else: 
            # We call the function `coordinates_consensus` to retrieve lat, lon (old method)
            lat, lon = self.coordinates_consensus()

        # We save this info in the dictionary
        self.island_info['spatial_reference']['latitude'] = float(lat)
        self.island_info['spatial_reference']['longitude'] = float(lon)

    def spectral_index_ndvi(self, image, geo_polygon, band1='B8', band2='B4'):
        """
        Calculate the NDVI (Normalised Difference Vegetation Index) for a given satellite image.

        Parameters:
        - image (ee.Image): The satellite image.
        - geo_polygon (ee.Geometry): The geometry representing the region of interest.
        - band1 (str, optional): The name of the first band for NDVI calculation.
        - band2 (str, optional): The name of the second band for NDVI calculation.

        Returns:
        - np.ndarray: An array of NDVI values.
        """
        # Get arrays from GEE image
        sample_rectangle = image.sampleRectangle(geo_polygon, defaultValue=1., defaultArrayValue=1.)
        arr_band1 = np.array(sample_rectangle.get(band1).getInfo())
        arr_band2 = np.array(sample_rectangle.get(band2).getInfo())

        # Equation for NDVI
        arr_ndvi = (arr_band1 - arr_band2)/(arr_band1 + arr_band2)

        return arr_ndvi

    def create_polygon(self, key, lat, lon, var):
        """
        Create an ee.Geometry.Polygon based on specified parameters.

        Parameters:
        - key (str): The key indicating the type of polygon ('top-left', 'top-right', etc.).
        - lat (float): Latitude coordinate.
        - lon (float): Longitude coordinate.
        - var (float): The variance used to create the polygon.

        Returns:
        - ee.Geometry.Polygon: The created polygon.
        """
        if key == 'top-left':
            geo_polygon = ee.Geometry.Polygon(
            [[[lon-var, lat+var],
                [lon-var, lat],
                [lon, lat],
                [lon, lat+var]]], None, False)

        elif key == 'top-right':
            geo_polygon = ee.Geometry.Polygon(
            [[[lon, lat+var],
                [lon, lat],
                [lon+var, lat],
                [lon+var, lat+var]]], None, False)

        elif key == 'bottom-left':
            geo_polygon = ee.Geometry.Polygon(
            [[[lon-var, lat],
                [lon-var, lat-var],
                [lon, lat-var],
                [lon, lat]]], None, False) 

        elif key == 'bottom-right':
            geo_polygon = ee.Geometry.Polygon(
            [[[lon, lat],
                [lon, lat-var],
                [lon+var, lat-var],
                [lon+var, lat]]], None, False)

        elif key == 'final':
            geo_polygon = ee.Geometry.Polygon(
            [[[lon-var, lat+var],
                [lon-var, lat-var],
                [lon+var, lat-var],
                [lon+var, lat+var]]], None, False)
        
        return geo_polygon

    def calculate_geometry_using_spectral_indices(self):
        """
        Calculate island geometry using spectral indices from satellite images.

        Returns:
        - ee.Geometry.Polygon: The calculated island geometry.
        """
        # Extract dict_geometry
        var_init = self.dict_geometry['var_init']
        var_change = self.dict_geometry['var_change']
        thresold_NDVI = self.dict_geometry['thresold_ndvi']
        var_limit = self.dict_geometry['var_limit']

        # Specify a geometric point in GEE
        lat = float(self.island_info['spatial_reference']['latitude'])
        lon = float(self.island_info['spatial_reference']['longitude'])
        point = ee.Geometry.Point([lon, lat])

        # Availability – Sentinel-2
        collection_S2 = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(point)

        # Pick cloud-free image
        image = collection_S2.sort('CLOUDY_PIXEL_PERCENTAGE').first()

        dict_coords = {"top-left": ['tl', 0, 0],
                       "top-right": ['tr', 0, -1],
                       "bottom-left": ['bl', -1, 0],
                       "bottom-right": ['br', -1, -1]}

        # Initial conditions for while loop
        var = var_init

        # While loop
        while var < var_limit:
            bool_borders = []
            bool_full = []
            if self.verbose: 
                print('var = ', var)

            # For loop for every 'corner'
            for key in dict_coords.keys():               
                # Generate polygon for a given 'corner'
                geo_polygon = self.create_polygon(key, lat, lon, var)

                # Calculate NDVI for every pixel of that 'corner'
                arr_NDVI = self.spectral_index_ndvi(image, geo_polygon)

                # Create array with the NDVI values on the border of the 'corner'
                arr_borders = np.concatenate((arr_NDVI[dict_coords[key][1], :], arr_NDVI[:, dict_coords[key][2]]))

                # Test conditions
                bool_borders.append(np.any(arr_borders > thresold_NDVI))
                bool_full.append(np.all(arr_NDVI < thresold_NDVI))

            # If we reach the limit, the loop ends and we calculate the region as is
            if (var >= (var_limit - var_change)):  
                if self.verbose: 
                     print('Maximum limit reached, code will stop')

                geo_polygon_final = self.create_polygon(key='final', lat=lat, lon=lon, var=var)
                
                return geo_polygon_final

            # Still land on the borders -> we expand the region at the next iteration
            if np.any(np.array(bool_borders)):
                var += var_change

            # Only water -> no island -> we expand the region at the next iteration
            elif np.all(np.array(bool_full)):
                var += var_change

            # We found an island surrounded by water -> loop ends
            else:
                var += var_change
                if self.verbose: 
                    print('Done!')

                geo_polygon_final = self.create_polygon(key='final', lat=lat, lon=lon, var=var)
                
                return geo_polygon_final

    def calculate_geometry(self):
        """
        Calculate island geometry using either OpenStreetMap or spectral indices.

        Updates the island_info dictionary with the island's polygon.
        """
        if self.polygon is not None:
            self.island_info['spatial_reference']['polygon'] = ee.Geometry.Polygon(self.polygon, None, False)
            return

        # New method (OpenStreetMap)
        if self.method == 'new':
            try:
                area = ox.geocode_to_gdf(self.island + ', ' + self.country)

                polygon_square_OSM = ee.Geometry.Polygon(
                [[[area.bbox_west[0]-0.001, area.bbox_north[0]+0.001],
                    [area.bbox_west[0]-0.001, area.bbox_south[0]-0.001],
                    [area.bbox_east[0]+0.001, area.bbox_south[0]-0.001],
                    [area.bbox_east[0]+0.001, area.bbox_north[0]+0.001],
                    [area.bbox_west[0]-0.001, area.bbox_north[0]+0.001]]], None, False)      

                polygon_OSM_geojson = geojson.Polygon(list((area.geometry[0].exterior.coords)))
                polygon_OSM = ee.Geometry.Polygon(polygon_OSM_geojson['coordinates']) 

                self.island_info['spatial_reference']['polygon'] = polygon_square_OSM
                self.island_info['spatial_reference']['polygon_OSM'] = polygon_OSM

            # Old method
            except:               
                polygon_old_method = self.calculate_geometry_using_spectral_indices()
                self.island_info['spatial_reference']['polygon'] = polygon_old_method

        # Old method
        else:
            polygon_old_method = self.calculate_geometry_using_spectral_indices()
            self.island_info['spatial_reference']['polygon'] = polygon_old_method

    def get_satellite_availability(self):
        """
        Retrieve satellite availability data for the island.

        Filters and stores satellite image collections based on specified criteria.
        """
        # Open satellite information from file
        dict_satellite_file = os.path.join(self.dict_satellite_path, 'dict_satellite.data')
        with open(dict_satellite_file, 'rb') as fw:
            dict_satellite = pickle.load(fw)

        # Create geometry point
        point = ee.Geometry.Point([self.island_info['spatial_reference']['longitude'], self.island_info['spatial_reference']['latitude']])

        # Create date range
        if self.date_range_user is None and self.list_sat_user is None:
            date_range = [dict_satellite['L5']['SR dataset availability'][0], datetime.datetime.now()]

        elif self.date_range_user is None and self.list_sat_user is not None:
            novelty_index_sat = [dict_satellite[sat]['novelty index'] for sat in self.list_sat_user]
            min_idx, max_idx = novelty_index_sat.index(min(novelty_index_sat)), novelty_index_sat.index(max(novelty_index_sat))
            date_range = [dict_satellite[self.list_sat_user[min_idx]]['SR dataset availability'][0], dict_satellite[self.list_sat_user[max_idx]]['SR dataset availability'][1]]

        else: 
            date_range = self.date_range_user
                
        # Create satellite list
        if self.list_sat_user is None: 
            list_sat = list(dict_satellite.keys())

        else: 
            list_sat = self.list_sat_user

        # Create empty dictionary for filtered ImageCollection
        image_collection_dict = {'description': 'Filtered (cloud threshold of {}%) ImageCollection for satellites of interest'.format(self.cloud_threshold)}

        # Loop in the list of satellites
        for sat in list_sat:
            collection_sat = ee.ImageCollection(dict_satellite[sat]['{} GEE Snipper'.format(self.image_type)]) \
                                .filterDate(date_range[0], date_range[1]) \
                                .filterBounds(point) \
                                .filterMetadata(dict_satellite[sat]['cloud label'], 'less_than', self.cloud_threshold)

            size_collection_sat = collection_sat.size().getInfo()

            if self.verbose: 
                print(sat, size_collection_sat)

            if size_collection_sat > 0:
                image_collection_dict[sat] = collection_sat

        self.island_info['image_collection_dict'] = image_collection_dict
    
    def get_other_info(self):
        """
        Retrieve additional information about the island from various sources (e.g., OpenStreetMap, Wikidata).
        """
        place = self.island + ', ' + self.country

        try:
            osm_type, osm_id = ox.geocode_to_gdf(place).osm_type.values[0], ox.geocode_to_gdf(place).osm_id.values[0]
        
        except:
            print('No other information available.')
            return

        if self.wikidata_id is None:
            url_other = 'https://www.openstreetmap.org/{}/{}'.format(osm_type, str(osm_id))
            web_url_other = urllib.request.urlopen(url_other)
            data_url_other = str(web_url_other.read())

            # Find wikidata ID
            pattern_wikidata_id = r'Q\d+'
            matches = re.findall(pattern_wikidata_id, data_url_other)
            if len(matches) > 0:
                wikidata_id = np.unique(np.array(matches))[0]
            
            else:
                print('No other information available.')
                return

        else:
            wikidata_id = self.wikidata_id

        # Fetch information about the Wikidata item
        wikidata = wdi_core.WDItemEngine(wd_item_id=wikidata_id).get_wd_json_representation()

        # List of properties to retrieve
        list_properties = self.relevant_properties_wikidata

        # Loop through the properties
        for prop in list_properties:            
            if prop in wikidata['claims'].keys():
                claim_prop = wikidata['claims'][prop]
                name_prop = wdi_core.WDItemEngine(wd_item_id=prop).get_label()

                if len(claim_prop) == 1:
                    try:
                        self.island_info['general_info'][name_prop] = wdi_core.WDItemEngine(wd_item_id=claim_prop[0]['mainsnak']['datavalue']['value']['id']).get_label()
                    except:
                        self.island_info['general_info'][name_prop] = claim_prop[0]['mainsnak']['datavalue']['value']['amount']

                else:
                    try:
                        self.island_info['general_info'][name_prop] = ', '.join([wdi_core.WDItemEngine(wd_item_id=claim_prop[lll]['mainsnak']['datavalue']['value']['id']).get_label() for lll in range(len(claim_prop))])

                    except:
                        self.island_info['general_info'][name_prop] = ', '.join([claim_prop[lll]['mainsnak']['datavalue']['value']['amount'] for lll in range(len(claim_prop))])
                        
            else:
                continue

    def create_dictionary_from_duvat_magnan_2019(self, idx, duvat_magnan_2019_df):
        """
        Create a dictionary to store information from the Duvat & Magnan (2019) dataset.

        Parameters:
        - idx (int): The index of the island's data in the dataset.
        - duvat_magnan_2019_df (pd.DataFrame): The Duvat & Magnan (2019) dataset.

        Updates the island_info dictionary with relevant data.
        """        
        # Retrieve the values of the island
        columns_values = duvat_magnan_2019_df.iloc[idx].values

        # Keep only the values that are not NaN
        idx_columns_not_nan = np.argwhere(~pd.isna(columns_values)).flatten()

        # Create dictionary for Duvat & Magnan (2019) information
        self.island_info['duvat_magnan_2019'] = {}

        # Loop through the values
        for column in idx_columns_not_nan:
            # Retrieve the value
            tuple_headers = duvat_magnan_2019_df.columns[column]

            # Create empty string for the dictionary key
            dict_key = ''

            # Loop in every element of the tuple
            for idx_tuple, element in enumerate(tuple_headers):
                if ~np.char.startswith(element, 'Unnamed'):
                    if idx_tuple == 0:
                        dict_key += element.replace('\n', '').lower()
                    else:
                        dict_key += ' ' + element.replace('\n', '').lower()
            
            # Save information in the dictionary
            if type(columns_values[column]) == int: 
                self.island_info['duvat_magnan_2019'][dict_key] = float(columns_values[column])

            elif dict_key == 'atoll':
                self.island_info['duvat_magnan_2019'][dict_key] = ' '.join(columns_values[column].split(' ')[1:])
                self.island_info['general_info'][dict_key] = ' '.join(columns_values[column].split(' ')[1:])
                
            elif dict_key == 'island':
                continue

            else:
                self.island_info['duvat_magnan_2019'][dict_key] = columns_values[column]

    def get_duvat_magnan_2019(self):
        """
        Retrieve information about the island from the Duvat & Magnan (2019) dataset.

        This method is specific to islands in the Maldives.
        """
        print('~ Retrieving information from Duvat & Magnan (2019). ~')

        # Open file with Duvat & Magnan (2019) data
        duvat_magnan_2019_file = os.path.join(self.duvat_magnan_2019_path, 'duvat_magnan_2019.xlsx')
        duvat_magnan_2019_df = pd.read_excel(duvat_magnan_2019_file, sheet_name='ALL ISLANDS_Full Database', skiprows=1, header = [0, 1, 2, 3]).iloc[:608]

        # Retrieve the row corresponding to the island
        idx_island = np.argwhere(duvat_magnan_2019_df[duvat_magnan_2019_df.columns[1]] == self.island).flatten()

        # Only one corresponding island
        if idx_island.size == 1:
            # Create dictionary
            self.create_dictionary_from_duvat_magnan_2019(idx_island[0], duvat_magnan_2019_df)
        
        elif idx_island.size > 1:
            print('More than one island with the same name. Please specify the island number.')
            print('Island numbers:', duvat_magnan_2019_df.iloc[idx_island]['Island number'].values.flatten())
            island_number = int(input('Please enter the number of the island you are interested in: ')) 

            # Retrieve the row corresponding to the island
            idx_island = np.argwhere(duvat_magnan_2019_df[duvat_magnan_2019_df.columns[2]] == island_number).flatten()  

            # Create dictionary
            self.create_dictionary_from_duvat_magnan_2019(idx_island[0], duvat_magnan_2019_df)   

        elif idx_island.size == 0:
            print('Island not found in the database.')
            return

    def main(self):
        """
        Main method for retrieving general and spatial information about the island.

        Orchestrates the execution of various steps and saves the collected information.
        """        
        print('\n-------------------------------------------------------------------')
        print('Retrieving general and spatial information about the island')
        print('Island:', ', '.join([self.island, self.country]))
        print('-------------------------------------------------------------------\n')

        # If the path in which the data will be stored doesn't exist, we create it
        if not os.path.exists(self.island_info_path): 
            os.makedirs(self.island_info_path)

        # Check what information is already available
        info_file_path = os.path.join(self.island_info_path, 'info_{}_{}.data'.format(self.island, self.country))
        
        if os.path.isfile(info_file_path):
            # Load the .data file with pickle
            with open(info_file_path, 'rb') as f:
                self.island_info = pickle.load(f)

            # If actions are defined by the user, skip to steps
            if self.to_do is None:       
                self.to_do = {'coordinates': True, 'polygon': True, 'availability': True, 'other': True, 'duvat_magnan_2019': True}

                # 1. Coordinates
                if self.island_info['spatial_reference']['latitude'] is not None and \
                    self.island_info['spatial_reference']['longitude'] is not None: 
                    self.to_do['coordinates'] = False

                # 2. Geometry
                if self.island_info['spatial_reference']['polygon'] is not None: 
                    self.to_do['polygon'] = False

                # 3. Image availability
                if self.island_info['image_collection_dict'] is not None: 
                    self.to_do['availability'] = False

                # 4. Other information
                if len(self.island_info['general_info']) > 3:
                    self.to_do['other'] = False

                # 5. information from Duvat & Magnan (2019) - Maldives only
                if 'duvat_magnan_2019' in self.island_info.keys() or self.country != 'Maldives':
                    self.to_do['duvat_magnan_2019'] = False

                # If all available information is already available, return dictionary
                if not any(self.to_do.values()):
                    print('~ All information is already available, returning information. ~')
                    return self.island_info

                else: 
                    print('~ The following information will be extracted/calculated:', \
                          ' and '.join([key for (key, val) in self.to_do.items() if val]), '~')                
            else: 
                print('~ The user wishes to extract/calculate', ' and '.join([key for (key, val) in self.to_do.items() if val]), '~')

        elif self.overwrite:
            self.to_do = {'coordinates': True, 'polygon': True, 'availability': True, 'other': True, 'duvat_magnan_2019': True}

        # No file exists
        else: 
            print('~ All information will be extracted/calculated. ~')
            self.island_info = {'general_info': {'island': self.island, 'country': self.country}, \
                        'spatial_reference': {'latitude': None, 'longitude': None, 'polygon': None}, \
                        'image_collection_dict': None}
            
            if self.alt_name is not None:
                self.island_info['general_info']['alt_name'] = self.alt_name

            self.to_do = {'coordinates': True, 'polygon': True, 'availability': True, 'other': True, 'duvat_magnan_2019': True}

            if self.country == 'Maldives':
                self.to_do['duvat_magnan_2019'] = True
        
        # (Re-)Calculating missing information
        for missing_info in [key for (key, val) in self.to_do.items() if val]:
            # Step 1: Extract coordinates (latitude, longitude)
            if missing_info == 'coordinates' or self.overwrite:
                self.retrieve_coordinates()

            # Step 2: Calculate geometry (ee.Geometry.Polygon)
            elif missing_info == 'polygon' or self.overwrite:
                self.calculate_geometry()

            # Step 3: Build a dictionary with satellite availability
            elif missing_info == 'availability' or self.overwrite:
                self.get_satellite_availability()

            # Step 4: Get other information (e.g. elevation, population, etc.)
            elif missing_info == 'other' or self.overwrite:
                self.get_other_info()

            # Step 4: Get other information (e.g. elevation, population, etc.)
            elif (missing_info == 'duvat_magnan_2019' or self.overwrite) and self.country == 'Maldives':
                try:
                    self.get_duvat_magnan_2019()
                
                except:
                    continue
        
        if self.atoll is not None:
            self.island_info['general_info']['atoll'] = self.atoll

        # Save dictionary
        with open(info_file_path, 'wb') as f:
            pickle.dump(self.island_info, f)

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TimeSeriesNOAACRW(IslandTimeBase):
    """
    Retrieves and processes time series data for sea surface temperature (SST)
    from NOAA Coral Reef Watch (CRW) for a specified island and country.

    Args:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): If True, print initialisation information. Default is True.
        overwrite (bool, optional): If True, overwrite existing data. Default is False.
    """
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.acronym = 'CRW'
        self.name_description = 'Sea surface temperature (NOAA Coral Reef Watch)'
        self.description = 'This module retrieves time series of sea surface temperature (SST) from NOAA Coral Reef Watch (CRW) for a given region.'
        self.description_timeseries = 'Daily mean SST for a 5km virtual region.'
        self.source = 'https://coralreefwatch.noaa.gov/product/vs/data.php'

    def get_available_stations(self):
        """
        Fetches a list of available stations from the NOAA Coral Reef Watch (CRW) website
        and stores them in the 'station_names' attribute.
        """
        # URL with all >200 stations
        url_CRW = 'https://coralreefwatch.noaa.gov/product/vs/data.php'
        web_CRW = urllib.request.urlopen(url_CRW)
        data_CRW = str(web_CRW.read())

        # Regex pattern for station names
        pattern_CRW = r'_ts_(\w+).png"\>'

        # Find all corresponding patterns
        station_names = re.findall(pattern_CRW, data_CRW)

        # Clean data
        station_names = np.unique(np.array(station_names))
        station_names = station_names[np.argwhere(~np.char.startswith(station_names, 'multiyr')).T[0]]

        # Keep data stored
        self.station_names = station_names
    
    def find_corresponding_station(self):
        """
        Attempts to find a corresponding station based on island and country names
        and stores it in the 'station_name' attribute.
        """
        possible_patterns = [self.island.lower().replace(' ', '_'), self.country.lower(), '_'.join((self.island.lower(), self.country.lower()))]
        self.station_to_retrieve = False

        for pp in possible_patterns:
            if pp in self.station_names:
                self.station_name = self.station_names[pp == self.station_names][0]
                self.station_to_retrieve = True

    def get_timeseries(self):
        """
        Retrieves and processes time series data for sea surface temperature (SST)
        for the specified island and country.
        """
        # Retrieve available stations
        self.get_available_stations()

        # Find station to retrieve
        self.find_corresponding_station()

        # If there is a station available
        if self.station_to_retrieve:
            print('~ The station to retrieve is `{}`. ~'.format(self.station_name))

            # Retrieve corresponding URL
            url_station = 'https://coralreefwatch.noaa.gov/product/vs/data/{}.txt'.format(self.station_name)
            web_station = urllib.request.urlopen(url_station)
            data_station = str(web_station.read())

            # Select relevant information
            station_arr_splitted = np.array(data_station.split('\\n'))
            idx_start = np.argwhere(np.char.startswith(station_arr_splitted, 'YYYY'))[0][0]
            station_arr = station_arr_splitted[idx_start:-1]

            # Clean data
            for idx_station in range(1, len(station_arr)): 
                if idx_station == 1:
                    station_arr_cleaned = np.array([item for item in station_arr[idx_station].split(' ') if item != ''], dtype=float)
                else:
                    station_arr_cleaned = np.row_stack((station_arr_cleaned, np.array([item for item in station_arr[idx_station].split(' ') if item != ''], dtype=float)))

            # Create DataFrame
            station_df = pd.DataFrame(station_arr_cleaned, columns = np.array(station_arr[0].split(' '), dtype=str))
            station_df['datetime'] = [datetime.datetime(year=int(station_df.YYYY[idx]), month=int(station_df.MM[idx]), day=int(station_df.DD[idx])) for idx in station_df.index]
            station_df['sea_surface_temperature_NOAACRW'] = [(station_df.SST_MIN[idx] + station_df.SST_MAX[idx])/2 for idx in station_df.index]
            station_df_final = station_df[['datetime', 'sea_surface_temperature_NOAACRW']].set_index('datetime').apply(pd.to_numeric)

            # Save information in dictionary
            self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = station_df_final
        
        # If there is no station available
        else:
            print('~ There is no station available for this island. ~')

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TimeSeriesNighttimeLight(IslandTimeBase):
    """
    Calculates time series of nighttime light (NTL) from remote sensing (DMSP-OLS)
    for a given island.

    Args:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): If True, print initialisation information. Default is True.
        overwrite (bool, optional): If True, overwrite existing data. Default is False.
    """
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.acronym = 'nighttime_light'
        self.name_description = 'Nighttime light (DMSP-OLS)'
        self.description = 'This module calculates time series of nighttime light (NTL) from remote sensing (DMSP-OLS) for a given island.'
        self.description_timeseries = 'Consistent and corrected nighttime night intensity time series.'
        self.source = 'Zhao, Cao, Chen & Cui (2020). A Consistent and Corrected Nighttime Light dataset (CCNL 1992-2013) from DMSP-OLS data (Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6644980. GEE: https://developers.google.com/earth-engine/datasets/catalog/BNU_FGS_CCNL_v1#citations'
    
    def get_timeseries(self):
        """
        Retrieves and calculates the time series of nighttime light (NTL) for the specified island.
        """
        def reduceRegionMean(img):
            """
            Calculates the mean of an image using Earth Engine's reducer and sets date and mean as properties.

            Args:
                img (ee.Image): An Earth Engine image.

            Returns:
                ee.Image: An Earth Engine image with date and mean properties set.
            """
            # Calculate mean with ee.Reducer
            img_mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=polygon, scale=30).get('b1')

            return img.set('date', img.date().format()).set('mean', img_mean)

        # Retrieve NTL collection from GEE
        collection_NTL = ee.ImageCollection("BNU/FGS/CCNL/v1")

        # Retrieve information from dictionary or inputs
        polygon = self.island_info['spatial_reference']['polygon']

        print('~ Retrieving NTL time series. ~')

        # Filter bounds and dates, select information
        collection = collection_NTL.filterBounds(polygon)

        # Take mean of the region
        collection_mean = collection.map(reduceRegionMean)

        # Create list with information
        nested_list = collection_mean.reduceColumns(ee.Reducer.toList(2), ['date', 'mean']).values().get(0)

        # Create pandas.DataFrame
        df_NTL = pd.DataFrame(nested_list.getInfo(), columns=['datetime', 'nighttime_light'])

        # Convert to date to datetime
        df_NTL['datetime'] = pd.to_datetime(df_NTL['datetime'])
        df_NTL = df_NTL.set_index('datetime')
        
        # If DataFrame is not empty (0)
        if all([el == 0 for el in df_NTL.nighttime_light]):
            print("Time series is empty!")
        
        else:
            # Save information in dictionary
            df_NTL = df_NTL.apply(pd.to_numeric)
            self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = df_NTL

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TimeSeriesDisasters(IslandTimeBase):
    """
    Retrieves and processes time series data for disasters from the Emergency Events Database (EM-DAT)
    for a specified island.

    Args:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): If True, print initialisation information. Default is True.
        overwrite (bool, optional): If True, overwrite existing data. Default is False.
    """
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.disasters_path = os.path.join(os.getcwd(), 'data', 'disasters')
        self.acronym = 'disasters'
        self.name_description = 'Disasters (EM-DAT)'
        self.description = 'The International Disaster Database, also called Emergency Events Database (EM-DAT), contains essential core data on the occurrence and effects of over 22,000 mass disasters in the world from 1900 to the present day. The database is compiled from various sources, including UN agencies, non-governmental organisations, insurance companies, research institutes and press agencies.'
        self.description_timeseries = 'timeseries_natural -> number of natural disasters per year | timeseries_technological -> number of technological disasters per year | timeseries_natural_cumulative -> cumulative number of natural disasters | timeseries_technological_cumulative -> cumulative number of technological disasters'
        self.source = 'https://www.emdat.be/'

    def get_timeseries(self):
        """
        Retrieves and processes time series data for disasters from the EM-DAT database.
        """
        print('~ Retrieving disaster data from EM-DAT database. ~')

        # Retrieve data from Excel file
        disasters_data = pd.read_excel(os.path.join(self.disasters_path, 'disasters_database_1975_2023.xlsx'))

        # Get DataFrame for the island
        disasters_data_island = disasters_data[disasters_data.Country == self.country]

        # Save information in dictionary
        self.island_info['timeseries_{}'.format(self.acronym)]['database'] = disasters_data_island

        # TIME SERIES NUMBER OF EVENTS PER YEAR

        print('~ Retrieving time series as number of events per year. ~')

        # Create column for `datetime`
        arr_datetime = np.array([datetime.datetime(year=year, month=1, day=1) \
                        for year in range(disasters_data_island.Year.values[0], \
                                          disasters_data_island.Year.values[-1] + 1)])
        
        # Create empty columns for `natural` and `technological`
        arr_natural = np.zeros_like(arr_datetime)
        arr_technological = np.zeros_like(arr_datetime)

        # Fill number of events per year
        for idx in disasters_data_island.index:
            if disasters_data_island['Disaster Group'][idx] == "Natural": 
                arr_natural[np.argwhere(arr_datetime == datetime.datetime(year=disasters_data_island['Year'][idx], month=1, day=1))[0][0]] += 1

            elif disasters_data_island['Disaster Group'][idx] == "Technological":
                arr_technological[np.argwhere(arr_datetime == datetime.datetime(year=disasters_data_island['Year'][idx], month=1, day=1))[0][0]] += 1

        # Calculate cumulative sum
        arr_natural_cumulative = np.cumsum(arr_natural)
        arr_technological_cumulative = np.cumsum(arr_technological)

        # Create multiple pd.DataFrame
        timeseries_disasters = pd.DataFrame(np.column_stack((arr_datetime, arr_natural, arr_technological, arr_natural_cumulative, arr_technological_cumulative)), \
                                          columns=['datetime', 'natural', 'technological', 'natural_cumulative', 'technological_cumulative'])
        timeseries_disasters = timeseries_disasters.set_index('datetime')
        timeseries_disasters = timeseries_disasters.apply(pd.to_numeric)

        # Save information in dictionary
        self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = timeseries_disasters

        # TIME SERIES CONFOUNDERS

        print('~ Retrieving time series as confounder effects. ~')

        # Create a daily range of dates
        daily_range_datetime = daily_range_datetime = pd.date_range(start=datetime.datetime(1975, 1, 1), end=datetime.datetime.now(), freq='D')

        # Initialise DataFrame for confounders
        col_names = [f'{row["Disaster Type"].replace(" ", "_").lower()}_{int(row["Year"])}' for idx, row in disasters_data_island.iterrows()]
        df_confounders = pd.DataFrame({'datetime': daily_range_datetime}).set_index('datetime')

        # List of geographical areas
        list_geo_locations = ', '.join([self.island_info['general_info'][key_info] for key_info in self.island_info['general_info'].keys()]).split(', ')

        # Create empty dictionary
        dict_disasters_confounders = {'datetime': daily_range_datetime}

        # Create DataFrame
        df_confounders = pd.DataFrame(dict_disasters_confounders).set_index('datetime')

        # Loop through catastrophes
        for it, idx in enumerate(disasters_data_island.index):
            row_event = disasters_data_island.loc[idx]

            if not pd.isnull(row_event['Location']) or not pd.isnull(row_event['Geo Locations']):
                list_location = [list_geo_locations[i] in [row_event['Location'].split(', ') if not pd.isnull(row_event['Location']) else []][0] for i in range(len(list_geo_locations))]
                list_geo_locations = [list_geo_locations[i] in [row_event['Geo Locations'].split(', ') if not pd.isnull(row_event['Geo Locations']) else []][0] for i in range(len(list_geo_locations))]
                list_t = np.array(list_location + list_geo_locations)

                if row_event['Location'] == 'Widespread': 
                    list_t = [True]

            else: 
                list_t = [True]

            # If the event relates to the island
            if np.any(list_t): 
                start_event = datetime.datetime(year=int(row_event['Start Year']), month=int([row_event['End Month'] if not np.isnan(row_event['End Month']) else 1][0]), day=int([row_event['Start Day'] if not np.isnan(row_event['Start Day']) else 1][0]))
                end_event = datetime.datetime(year=int(row_event['End Year']), month=int([row_event['End Month'] if not np.isnan(row_event['End Month']) else 1][0]), day=int([row_event['End Day'] if not np.isnan(row_event['End Day']) else 1][0]))
                df_confounders[col_names[it]] = np.zeros(len(daily_range_datetime))
                df_confounders[col_names[it]][(df_confounders.index >= start_event) & (df_confounders.index <= end_event)] = 1

        # Save information in dictionary
        self.island_info['timeseries_{}'.format(self.acronym)]['confounders'] = df_confounders

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TimeSeriesWorldBank(IslandTimeBase):
    """
    Retrieves and processes time series data from the World Bank for a specified island.

    Args:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): If True, print initialisation information. Default is True.
        overwrite (bool, optional): If True, overwrite existing data. Default is False.
    """
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.acronym = 'WorldBank'
        self.name_description = 'World Bank'
        self.description = 'Free and open access to global development data.'
        self.description_timeseries = 'Yearly time series of socioeconomic indicators.'
        self.source = 'https://data.worldbank.org/'

    def find_country_ID(self):
        """
        Find the World Bank country ID based on the provided country name.

        Returns:
            str: The World Bank country ID.
        """
        # Query World Bank (economy section)
        wb_featureset = wb.economy.info(q=self.country)

        # Find ID in FeatureSet
        country_ID = wb_featureset.items[0]['id']

        return country_ID

    def get_timeseries(self):
        """
        Retrieves time series data from the World Bank and saves it in the island's information dictionary.
        """        
        # Retrieve information for the whole country
        island_info_other_island = super().other_info_in_island_info_path()

        if island_info_other_island is not None:
            self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = island_info_other_island

        else:
            print('~ Retrieving time series. ~')

            # Retrieve country ID
            if not 'country_ID' in self.island_info['general_info'].keys():
                country_ID = self.find_country_ID()

                # Save in dictionary
                self.island_info['general_info']['country_ID'] = country_ID
            
            else:
                country_ID = self.island_info['general_info']['country_ID']

            # Query World Bank Database (series -> actual data)
            series_info = wb.series.info()

            # Create a pandas.DataFrame with all the information available to retrieve
            df_series_info = pd.DataFrame(vars(series_info).get('items'))

            # Add description of this timeseries
            self.island_info['timeseries_{}'.format(self.acronym)]['extensive_description_timeseries'] = df_series_info.set_index('id').to_dict()['value']

            # Retrieve DataFrame for this country
            df_WBD = wb.data.DataFrame(list(df_series_info.id), country_ID).T

            # Manage datetime and indices
            df_WBD['datetime'] = [datetime.datetime(year=int(df_WBD.index[idx].replace("YR", "")), month=1, day=1) for idx in range(len(df_WBD.index))]
            df_WBD = df_WBD.set_index('datetime')

            # Drop NaN values (no data at all) and constant values
            df_WBD = df_WBD.dropna(axis=1, how='all').drop(columns=df_WBD.columns[df_WBD.nunique() == 1]).apply(pd.to_numeric)

            # Save information in dictionary
            self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = df_WBD

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TimeSeriesWHO(IslandTimeBase):
    """
    Retrieves and processes time series data from the World Health Organization (WHO) for a specified island.

    Args:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): If True, print initialisation information. Default is True.
        overwrite (bool, optional): If True, overwrite existing data. Default is False.
    """
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.WHO_path = os.path.join(os.getcwd(), 'data', 'WHO')
        self.acronym = 'WHO'
        self.name_description = 'World Health Organization (WHO)'
        self.description = 'This module allows us to retrieve socioeconomics and environmental data from World Health Organization (WHO).'
        self.description_timeseries = 'Yearly time series of socioeconomic and environmental indicators.'
        self.source = 'https://www.who.int/data'

    def get_indicators(self):
        """
        Retrieves WHO indicators and saves them in the island's information dictionary.
        """
        file_indicators_WHO = os.path.join(self.WHO_path, 'indicators_df.xlsx')
        file_indicators_WHO_json = os.path.join(self.WHO_path, 'indicators.json')

        # Read DataFrame of indicators
        if os.path.exists(file_indicators_WHO):
            indicators_df = pd.read_excel(file_indicators_WHO) # load Excel file
        
        # If DataFrame does not exist, generate it from .json file
        else:
            indicators_json = json.load(open(file_indicators_WHO_json))['value'] # load json file

            # Create array for codes and descriptions
            codes = np.array([indicators_json[code]['IndicatorCode'] for code in range(len(indicators_json))])
            dess = np.array([indicators_json[name]['IndicatorName'] for name in range(len(indicators_json))])

            # Create combined array and DataFrame
            indicators_arr = np.column_stack((codes, dess))
            indicators_df = pd.DataFrame(indicators_arr, columns=['Code', 'Name'])

            # Save DataFrame
            indicators_df.to_excel(file_indicators_WHO, index=None)
        
        # Save information in dictionary
        self.island_info['timeseries_{}'.format(self.acronym)]['extensive_description_timeseries'] = indicators_df.set_index('Code').to_dict()['Name']
    
    def get_timeseries(self):
        """
        Retrieves time series data from the WHO and saves it in the island's information dictionary.
        """
        # Retrieve information for the whole country
        island_info_other_island = super().other_info_in_island_info_path()

        if island_info_other_island is not None:
            self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = island_info_other_island

        else:
            print('~ Retrieving time series. ~')

            if not 'country_ID' in self.island_info['general_info'].keys():
                country_ID = TimeSeriesWorldBank(self.island, self.country).find_country_id()

                # Save in dictionary
                self.island_info['general_info']['country_ID'] = country_ID

            else:
                country_ID = self.island_info['general_info']['country_ID'] # retrieve information from dictionary

            # Get Dataframe of indicators
            self.get_indicators()
            indicators_dict = self.island_info['timeseries_{}'.format(self.acronym)]['extensive_description_timeseries']

            # Other relevant information
            headers = {'Content-type': 'application/json'}

            idx_WHO = 0
            # Loop in every indicator
            for indicator in tqdm(list(indicators_dict.keys())):

                # Request information from WHO database
                post = requests.post('https://ghoapi.azureedge.net/api/{}'.format(indicator), headers=headers)
                
                # See if the URL is readable
                try:  
                    data = json.loads(post.text)
                    
                except: 
                    continue

                # Select values from dictionary
                data_list = data['value']

                # Make a DataFrame out of the data_list dictionary
                df_data_list = pd.DataFrame(data_list)

                # Select relevant columns and sort by year
                df_data_list_selection = df_data_list[(df_data_list.SpatialDim == country_ID) & ((df_data_list.Dim1 == 'BTSX') | (df_data_list.Dim1 == 'TOTL'))].sort_values(['TimeDim'])

                # Check if data is available
                if df_data_list_selection.shape[0] > 0:
                    dfs = df_data_list_selection[['TimeDim', 'NumericValue']].copy()

                    try:
                        dfs['datetime'] = [datetime.datetime(year=dfs.TimeDim[idx], month=1, day=1) for idx in dfs.index]

                    except:
                        continue
                    
                    dfs_rename = dfs[['datetime', 'NumericValue']].set_index('datetime').rename(columns={'NumericValue': indicator})
                    dfs_grouped = dfs_rename.groupby('datetime').mean()
                    
                    if idx_WHO == 0:
                        dfs_t = dfs_grouped

                    else:
                        dfs_t = pd.concat([dfs_t, dfs_grouped], axis=1)

                    idx_WHO += 1
                
            # Drop NaN values (no data at all) and constant values
            dfs_t = dfs_t.dropna(axis=1, how='all').drop(columns=dfs_t.columns[dfs_t.nunique() == 1]).apply(pd.to_numeric)

            # Save information in dictionary
            self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = dfs_t

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TimeSeriesSeaLevelAnomaly(IslandTimeBase):
    """
    Retrieves and calculates time series data of sea level anomaly from remote sensing (Copernicus) for a specified island.

    Args:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): If True, print initialisation information. Default is True.
        overwrite (bool, optional): If True, overwrite existing data. Default is False.
    """
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.copernicus_data_path = os.path.join(os.getcwd(), 'data', 'copernicus_data')
        self.acronym = 'sea_level_anomaly'
        self.name_description = 'Sea level anomaly'
        self.description = 'This module calculates time series of sea level anomaly from remote sensing (Copernicus) for a given island.'
        self.description_timeseries = 'Sea level anomaly in metres (m).'
        self.source = 'Copernicus Climate Change Service (C3S) (2017): ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate. Copernicus Climate Change Service Climate Data Store (CDS), (date of access), https://cds.climate.copernicus.eu/cdsapp#!/home'

    def get_timeseries(self):
        """
        Retrieve and calculate time series data of sea level anomaly and save it in the island's information dictionary.
        """
        if not 'units' in self.island_info['timeseries_{}'.format(self.acronym)].keys():
            self.island_info['timeseries_{}'.format(self.acronym)]['units'] = {'wave_energy': 'kW/m'}

        print('~ Retrieving time series. ~')

        # Retrieve lat/lon of interest
        latitude, longitude = self.island_info['spatial_reference']['latitude'], self.island_info['spatial_reference']['longitude']

        # Retrieve all Copernicus files (sea level gridded data)
        files_copernicus = os.listdir(self.copernicus_data_path)

        # Create empty lists
        date_list = []
        sla_list = []

        # Loop through all files
        for idx, file in enumerate(files_copernicus):
            if file.endswith('.nc'):
                # Open file as xarray dataset
                ds = xr.open_dataset(os.path.join(self.copernicus_data_path, file))
                time = ds['time']

                # Retrieve index for given lat/lon (one time only)
                if idx == 0:
                    # Find the index of the grid point nearest a specific lat/lon   
                    abslat = np.abs(ds.latitude - latitude)
                    abslon = np.abs(ds.longitude - longitude)
                    idx_grid = np.maximum(abslon, abslat)
                    ([xloc], [yloc]) = np.where(idx_grid == np.min(idx_grid))

                # Retrieve the closest grid to the islad
                point_ds = ds.isel({'longitude': xloc, 'latitude': yloc})

                # Append list with values
                sla_list.append(point_ds.sla.values[0])
                date_list.append(time.values[0])

        # Create DataFrame
        df_sea_level_anomaly = pd.DataFrame({'datetime': date_list, 'sea_level_anomaly': sla_list})
        df_sea_level_anomaly = df_sea_level_anomaly.set_index('datetime')
        
        # Save information in dictionary
        df_sea_level_anomaly = df_sea_level_anomaly.apply(pd.to_numeric)
        self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = df_sea_level_anomaly

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TimeSeriesPMLV2(IslandTimeBase):
    """
    Retrieves and calculates time series data of Penman-Monteith-Leuning Evapotranspiration V2 (PML_V2)
    from remote sensing for a specified island.

    Args:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): If True, print initialisation information. Default is True.
        overwrite (bool, optional): If True, overwrite existing data. Default is False.
    """
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.acronym = 'PMLV2'
        self.name_description = 'Penman-Monteith-Leuning Evapotranspiration V2 (PML_V2)'
        self.description = 'This module calculates time series of Gross primary product, Vegetation transpiration, Soil evaporation, Interception from vegetation canopy, Water body, snow and ice evaporation from remote sensing (PML_V2) for a given island.'
        self.description_timeseries = {'GPP': 'Gross primary product',
                                        'Ec': 'Vegetation transpiration',
                                        'Es': 'Soil evaporation',
                                        'Ei': 'Interception from vegetation canopy',
                                        'ET_water': 'Water body, snow and ice evaporation'}
        self.source = 'Penman-Monteith-Leuning Evapotranspiration V2 (PML_V2) products. GEE: https://developers.google.com/earth-engine/datasets/catalog/CAS_IGSNRR_PML_V2_v017#description'
        
    def get_timeseries(self):
        """
        Retrieve and calculate time series data of PML_V2 and save it in the island's information dictionary.
        """
        if not 'units' in self.island_info['timeseries_{}'.format(self.acronym)].keys():
            self.island_info['timeseries_{}'.format(self.acronym)]['units'] = {'GPP': 'gC m-2 d-1',
                                                                               'Ec': 'mm/d',
                                                                               'Es': 'mm/d',
                                                                               'Ei': 'mm/d',
                                                                               'ET_water': 'mm/d'}

        def reduceRegionMean(img):

            # Calculate mean with ee.Reducer
            img_mean = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=polygon, scale=30).get(info_PMLV2)

            return img.set('date', img.date().format()).set('mean', img_mean)

        # Retrieve PML_V2 collection from GEE
        collection_PML_V2 = ee.ImageCollection('CAS/IGSNRR/PML/V2_v017')

        # Retrieve information from dictionary or inputs
        polygon = self.island_info['spatial_reference']['polygon']

        # List of informations to retrieve
        list_to_retrieve = list(self.island_info['timeseries_{}'.format(self.acronym)]['description_timeseries'].keys())

        idx_PMLV2 = 0

        # Loop in all information to retrieve
        for info_PMLV2 in list_to_retrieve:

            print('~ Retrieving {}. ~'.format(self.island_info['timeseries_{}'.format(self.acronym)]['description_timeseries'][info_PMLV2]))

            # Filter bounds and dates, select information
            collection = collection_PML_V2.filterBounds(polygon).select(info_PMLV2)

            # Take mean of the region
            collection_mean = collection.map(reduceRegionMean)

            # Create list with information
            nested_list = collection_mean.reduceColumns(ee.Reducer.toList(2), ['date', 'mean']).values().get(0)

            if np.shape(np.array(nested_list.getInfo()))[0] == 0:
                continue

            else:
                if idx_PMLV2 == 0:
                    df_PMLV2 = pd.DataFrame(nested_list.getInfo(), columns=['datetime', self.island_info['timeseries_{}'.format(self.acronym)]['description_timeseries'][info_PMLV2].replace(',', '').replace(' ', '_').lower()])
            
                else:
                    df_PMLV2[self.island_info['timeseries_{}'.format(self.acronym)]['description_timeseries'][info_PMLV2].replace(',', '').replace(' ', '_').lower()] = np.array(nested_list.getInfo())[:, 1].astype('float')

                idx_PMLV2 += 1

        if 'df_PMLV2' in locals():
            df_PMLV2['datetime'] = pd.to_datetime(df_PMLV2['datetime']).set_index('datetime', inplace=True).apply(pd.to_numeric) # convert to date to datetime
                
            if df_PMLV2.shape[0] > 0:
                self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = df_PMLV2 # save information in dictionary
        
        else:
            return

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TimeSeriesPSMSL(IslandTimeBase):
    """
    Retrieves tide-gauge sea-level data from the Permanent Service for Mean Sea Level (PSMSL) for a specified island.

    Args:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): If True, print initialisation information. Default is True.
        overwrite (bool, optional): If True, overwrite existing data. Default is False.
    """
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.number_of_stations = 3
        self.PSMSL_path = os.path.join(os.getcwd(), 'data', 'PSMSL')
        self.acronym = 'PSMSL'
        self.name_description = 'Permanent Service for Mean Sea Level (PSMSL)'
        self.description = 'This module allows us to retrieve tide-gauge sea-level data from the Permanent Service for Mean Sea Level (PSMSL).'
        self.description_timeseries = 'Sea-level time series from tide-gauge stations.'
        self.source = 'https://psmsl.org/data/obtaining/'

    def find_closest_stations(self):
        """
        Find the closest tide-gauge stations to the island and save their information.
        """
        # Read Excel find with all available stations 
        stations_PSMSL = pd.read_excel(os.path.join(self.PSMSL_path, 'stations_PSMSL.xlsx'))

        # Create list of points with lat/lon of the island and of the station
        coords_island = (self.island_info['spatial_reference']['latitude'], self.island_info['spatial_reference']['longitude'])
        coords_stations = [(stations_PSMSL.lat.values[idx], stations_PSMSL.lon.values[idx]) for idx in stations_PSMSL.index]

        # Calculate distance (in km) between the island and every station
        distance_island_stations = np.array([geopy.distance.geodesic(coords_island, coords_stations[idx]).km for idx in range(len(coords_stations))])

        # Sort indices and keep the first three stations
        closest_stations_idx = np.argsort(distance_island_stations)[:self.number_of_stations]
        closest_stations_PSMSL = stations_PSMSL.iloc[closest_stations_idx]
        closest_distance_island_stations = distance_island_stations[closest_stations_idx]

        # Clean DataFrame
        closest_stations_PSMSL = closest_stations_PSMSL.set_index('ID')[['Station_name', 'lat', 'lon', 'Country']].rename(columns={'Station_name': 'station_name', 'lat': 'station_latitude', 'lon': 'station_longitude', 'Country': 'country_ID'})
        closest_stations_PSMSL['distance_from_island'] = closest_distance_island_stations

        # Save for other functions
        self.closest_stations_PSMSL = closest_stations_PSMSL

        # Add description of this timeseries
        self.island_info['timeseries_{}'.format(self.acronym)]['stations'] = closest_stations_PSMSL

    def get_timeseries(self):
        """
        Retrieve tide-gauge sea-level data for the closest stations and save it in the island's information dictionary.
        """        
        # Find closest stations
        self.find_closest_stations()

        # String for order of stations
        order_stations = ['{}{}'.format(i+1, 'th' if i+1 > 3 else ['st', 'nd', 'rd'][i]) for i in range(self.number_of_stations)]

        for idx_PSMSL, station in enumerate(self.closest_stations_PSMSL.index):
            
            print('~ Retrieving data from the {} closest station:'.format(order_stations[idx_PSMSL]), \
                    self.closest_stations_PSMSL.loc[self.closest_stations_PSMSL.index[idx_PSMSL]]['station_name'], \
                    'at a distance of', np.round(self.closest_stations_PSMSL.loc[self.closest_stations_PSMSL.index[idx_PSMSL]]['distance_from_island'], 1), \
                    'km from the island. ~')

            # Retrieve data from corresponding URL
            url_station = 'https://psmsl.org/data/obtaining/rlr.monthly.data/{}.rlrdata'.format(station)
            web_station = urllib.request.urlopen(url_station)
            data_station = str(web_station.read())

            # Data cleaning
            for idx_line, line in enumerate(data_station.split('\\n')[:-1]):
                if idx_line == 0:
                    arr_station = np.array(line.replace('b', '').replace("'", '').strip().split(';'), dtype=float)              
                else:
                    arr_station = np.row_stack((arr_station, np.array(line.replace('b', '').replace("'", '').strip().split(';'), dtype=float)))

            # Replace extreme values by NaN
            arr_station[:, 1][arr_station[:, 1] == -99999] = np.nan

            # Create datetime array
            arr_datetime = np.array([datetime.datetime(int(arr_station[i, 0]), 1, 1) + datetime.timedelta(days=(arr_station[i, 0] - int(arr_station[i, 0])) * 365.25) for i in range(len(arr_station[:, 0]))])

            # Create DataFrame
            df_station = pd.DataFrame(np.column_stack((arr_datetime, arr_station[:, 1])), columns=['datetime', 'sea_level_{}'.format(self.closest_stations_PSMSL.loc[self.closest_stations_PSMSL.index[idx_PSMSL]]['station_name'])])
            df_station = df_station.set_index('datetime')

            # Concatenate information
            if idx_PSMSL == 0:
                df_PSMSL_total = df_station
            
            else:
                df_PSMSL_total = pd.concat([df_PSMSL_total, df_station], axis=1)

        # Save information in dictionary
        df_PSMSL_total = df_PSMSL_total.apply(pd.to_numeric)
        self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = df_PSMSL_total

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TimeSeriesPSLGM(IslandTimeBase):
    """
    Retrieves data from the Pacific Sea Level and Geodetic Monitoring Project (PSLGM) for a specified island.

    Args:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): If True, print initialisation information. Default is True.
        overwrite (bool, optional): If True, overwrite existing data. Default is False.
    """
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.PSLGM_path = os.path.join(os.getcwd(), 'data', 'PSLGM')
        self.acronym = 'PSLGM'
        self.name_description = 'Pacific Sea Level and Geodetic Monitoring Project (PSLGM)'
        self.description = 'This module allows us to retrieve data from the Pacific Sea Level and Geodetic Monitoring Project (Monthly Sea Level and Meteorological Statistics).'
        self.description_timeseries = 'Available information: Monthly sea level, barometric pressure, water temperature and air temperature.'
        self.source = 'http://www.bom.gov.au/oceanography/projects/spslcmp/data/monthly.shtml'

    def get_timeseries(self):
        """
        Retrieve sea-level, barometric pressure, water temperature, and air temperature data from the PSLGM and save it in the island's information dictionary.
        """
        for idx_PSLGM, info in enumerate(['sea_level', 'barometric_pressure', 'water_temperature', 'air_temperature']):
            try:
                # Read data from file
                data_PSLGM = np.array(open(os.path.join(self.PSLGM_path, '{}_{}_{}.txt'.format(self.island, \
                                                                        self.country, \
                                                                        info)), 'r').readlines())
                
            except: 
                print('No PSLGM information for this island.')
                return self.island_info

            print("~ Retrieving {}. ~".format(info.replace('_', ' ').capitalize()))

            # Select rows corresponding to the data
            data_cleaned = np.array(data_PSLGM[np.argwhere(np.char.startswith(data_PSLGM, '      Mth'))[0][0] + 1 : \
                                        np.argwhere(np.char.startswith(data_PSLGM, '      Totals'))[0][0]])

            for row in range(len(data_cleaned)):

                row_cleaned = np.array(data_cleaned[row].replace('\n', '').split(' '))
                row_fully_cleaned = row_cleaned[row_cleaned != ''].astype(float)

                # Expected length is 8 (otherwise there is missing data)
                if len(row_fully_cleaned) != 8: continue

                # First iteration
                if row == 0: full_data = row_fully_cleaned
                
                # Stack with existing data
                else: full_data = np.row_stack((full_data, row_fully_cleaned))
        
            # Create pandas.DataFrame
            df_PSLGM = pd.DataFrame(full_data, columns=['Month', 'Year', 'Gaps', 'Good', 'Minimum', 'Maximum', 'Mean', 'StD'])
            df_PSLGM['datetime'] = [datetime.datetime(year=int(df_PSLGM.Year[idx]), month=int(df_PSLGM.Month[idx]), day=1) for idx in df_PSLGM.index]
            df_PSLGM = df_PSLGM.set_index('datetime')
            df_PSLGM = df_PSLGM.rename(columns={'Mean': info})

            if idx_PSLGM == 0:
                df_PSLGM_total = df_PSLGM[[info]]
            
            else:
                df_PSLGM_total = pd.concat([df_PSLGM_total, df_PSLGM[info]], axis=1)

        # Save information in dictionary
        df_PSLGM_total = df_PSLGM_total.apply(pd.to_numeric)
        self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = df_PSLGM_total

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TimeSeriesCoastSat(IslandTimeBase):
    """
    A class for retrieving time series of coastline positions along transects using CoastSat.

    Attributes:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): Whether to display verbose initialisation messages. Default is True.
        overwrite (bool, optional): Whether to overwrite existing data. Default is False.
        date_range (list, optional): The date range for data retrieval. Default is ['1984-01-01', '2022-12-31'].
        sat_list (list, optional): The list of satellites to use. Default is ['L5', 'L7', 'L8', 'L9', 'S2'].
        collection (str, optional): The data collection to use. Default is 'C02'.
        plot_results (bool, optional): Whether to plot the results. Default is True.
        distance_between_transects (int, optional): The distance between transects in meters. Default is 100.
        length_transect (int, optional): The length of transects in meters. Default is 250.
        reference_shoreline_transects_only (bool, optional): Whether to use reference shoreline for transects only. Default is True.
    """

    def __init__(self, island, country, verbose_init=True, overwrite=False, date_range=['2010-01-01', '2022-12-31'], sat_list=['L8', 'L9', 'S2'], \
                 collection='C02', plot_results=False, distance_between_transects=50, length_transect=250, reference_shoreline_transects_only=False, \
                 extract_shorelines=True, re_download=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.date_range = date_range
        self.sat_list = sat_list
        self.collection = collection
        self.coastsat_data_path = os.path.join(os.getcwd(), 'data', 'coastsat_data')
        self.plot_results = plot_results
        self.distance_between_transects = distance_between_transects
        self.length_transect = length_transect
        self.reference_shoreline_transects_only = reference_shoreline_transects_only
        self.extract_shorelines = extract_shorelines
        self.re_download = re_download
        self.acronym = 'coastsat'
        self.name_description = 'CoastSat (coastline position along transects)'
        self.description = 'This module allows us to retrieve time series of coastline position along transects using CoastSat.'
        self.description_timeseries = 'Coastline position time series along transects.'
        self.source = 'https://github.com/kvos/CoastSat'

    def get_reference_shoreline(self, metadata, settings):
        """
        Retrieve the reference shoreline from OpenStreetMap or manually.

        Args:
            metadata: Metadata information.
            settings: Settings for shoreline extraction.

        Returns:
            np.ndarray: The reference shoreline coordinates.
        """
        print('~ Retrieving reference shoreline from OpenStreetMap or manually. ~')
        
        # Check if OSM data is available
        try:
            gdf_coastline = ox.features_from_place(', '.join([self.island, self.country]), {'natural': 'coastline'})
            print('OSM coastline data available for this island.')

        except:
            print('No OSM data available for this island, we will manually define the reference shoreline.')
            reference_shoreline = SDS_preprocess.get_reference_sl(metadata, settings)
        
            return reference_shoreline

        # Get geometry of the coastline
        gdf_coastline_geometry = np.array(gdf_coastline['geometry'])

        # Loop over the coordinates of the coastline
        for idx_coastline in range(len(gdf_coastline_geometry)):
            
            # If the geometry is a shapely.geometry.linestring.LineString
            if type(gdf_coastline_geometry[idx_coastline]) == shapely.geometry.linestring.LineString:
                x_coastline = np.array([gdf_coastline_geometry[idx_coastline].xy[0][i] for i in range(len(gdf_coastline_geometry[idx_coastline].xy[0]))])
                y_coastline = np.array([gdf_coastline_geometry[idx_coastline].xy[1][i] for i in range(len(gdf_coastline_geometry[idx_coastline].xy[1]))])

            # If the geometry is a shapely.geometry.polygon.Polygon
            elif type(gdf_coastline_geometry[idx_coastline]) == shapely.geometry.polygon.Polygon:
                x_coastline = np.array([gdf_coastline_geometry[idx_coastline].exterior.xy[0][i] for i in range(len(gdf_coastline_geometry[idx_coastline].exterior.xy[0]))])
                y_coastline = np.array([gdf_coastline_geometry[idx_coastline].exterior.xy[1][i] for i in range(len(gdf_coastline_geometry[idx_coastline].exterior.xy[1]))])                

            # Interpolate between points to fill the gaps
            for pts in range(len(x_coastline) - 1):

                # Create a new array of points between the two points
                xx_coastline = np.array([x_coastline[pts], x_coastline[pts+1]])
                yy_coastline = np.array([y_coastline[pts], y_coastline[pts+1]])

                # Interpolate between the two points
                xx_coastline_linspace = np.linspace(xx_coastline[0], xx_coastline[1], 300)
                interpolation = interpolate.interp1d(xx_coastline, yy_coastline)

                # If first pair of points
                if pts == 0:
                    xx_coastline_full = xx_coastline_linspace
                    yy_coastline_full = interpolation(xx_coastline_linspace)

                # Concatenate with previous results
                else:
                    xx_coastline_full = np.concatenate((xx_coastline_full, xx_coastline_linspace))
                    yy_coastline_full = np.concatenate((yy_coastline_full, interpolation(xx_coastline_linspace)))

            # If first iteration
            if idx_coastline == 0:
                x_coastline_all = xx_coastline_full
                y_coastline_all = yy_coastline_full
            
            # Concatenate with previous results
            else:
                x_coastline_all = np.concatenate((x_coastline_all, xx_coastline_full))
                y_coastline_all = np.concatenate((y_coastline_all, yy_coastline_full))

        # Define the source and target coordinate systems
        src_crs = pyproj.CRS('EPSG:4326')
        tgt_crs = pyproj.CRS('EPSG:3857')

        # Create a transformer
        transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

        # Reproject the data
        x_reprojected, y_reprojected = transformer.transform(x_coastline_all, y_coastline_all)

        # Combine data into a numpy array  
        reference_shoreline = np.column_stack((x_reprojected, y_reprojected))

        return reference_shoreline

    def get_transects(self):
        """
        Create transects separated by a specified distance.
        """
        print('~ Creating transects separated by {} m. ~'.format(self.distance_between_transects))

        # Retrieve reference shoreline
        reference_shoreline = self.island_info['spatial_reference']['reference_shoreline']

        # Double the coordinates of the reference shoreline (to ensure cyclic boundary conditions)
        reference_shoreline_cyclic = np.row_stack((np.flip(reference_shoreline, axis=0), reference_shoreline))

        # Create a shapely.geometry.Polygon with the reference shoreline
        polygon_shoreline = shapely.geometry.Polygon(reference_shoreline)

        # Create empty dictionary for transects
        transects = {}

        # Starting conditions for the loop
        idx_equidistant_transects = []
        start_point_transect = 0

        # Loop to find indices for equidistant transects
        while start_point_transect < (len(reference_shoreline) - 1):
            idx_equidistant_transects.append(start_point_transect)
            idx_distance = start_point_transect + 1

            # Calculate the distance between current and next points
            distance_between_points = np.hypot((reference_shoreline[start_point_transect, 0] - reference_shoreline[idx_distance, 0]),\
                                                (reference_shoreline[start_point_transect, 1] - reference_shoreline[idx_distance, 1]))
            
            # Continue to check distances until threshold is met
            while distance_between_points < self.distance_between_transects:
                idx_distance += 1

                if idx_distance >= (len(reference_shoreline) - 1):
                    break
                
                # Recalculate distance
                distance_between_points = np.hypot((reference_shoreline[start_point_transect, 0] - reference_shoreline[idx_distance, 0]),\
                                                    (reference_shoreline[start_point_transect, 1] - reference_shoreline[idx_distance, 1]))
    
            start_point_transect = idx_distance        
        
        # Convert list to array
        idx_equidistant_transects = np.array(idx_equidistant_transects)

        # Useful recurrent variables for the loop
        min_x = np.min(reference_shoreline_cyclic[:, 0])
        max_x = np.max(reference_shoreline_cyclic[:, 0])
        range_min_max = np.linspace(min_x - 0.5*(max_x - min_x), max_x + 0.5*(max_x - min_x), 5000)

        # Loop over the number of transects
        for idx_transect, idx_coords in enumerate(int(len(reference_shoreline_cyclic)/2) + idx_equidistant_transects):  

            # Get an array of 4 points around the coordinates of interest
            x_around = reference_shoreline_cyclic[(idx_coords-2):(idx_coords+4), 0]
            y_around = reference_shoreline_cyclic[(idx_coords-2):(idx_coords+4), 1]

            # Fit an affine function to the 4 points
            m_shoreline, _ = np.polyfit(x_around, y_around, 1)

            # Get index for middle point
            idx_middle_point = int(len(x_around)/2 - 0.5)

            # Get perpendicular affine function
            m_perpendicular = -1 / m_shoreline
            b_perpendicular = y_around[idx_middle_point] - m_perpendicular * x_around[idx_middle_point]

            # Create array of perpendicular coordinates (using the affine function)
            x_perpendicular = range_min_max
            y_perpendicular = m_perpendicular * x_perpendicular + b_perpendicular

            # Create an array of shapely.geometry.Point with the perpendicular coordinates
            points_perpendicular = [shapely.geometry.Point(x, y) for (x, y) in zip(x_perpendicular, y_perpendicular)]

            # Indices inside the polygon (island)
            idx_inside_polygon = np.argwhere(polygon_shoreline.contains(points_perpendicular)).flatten()

            if len(idx_inside_polygon) < 2:
                continue

            # Determine the direction of the perpendicular vector by calculating the distance between the middle point and the first and last point inside the polygon
            first_idx_inside_polygon, last_idx_inside_polygon = idx_inside_polygon[0], idx_inside_polygon[-1]
            dist_first_idx_inside_polygon = np.hypot(x_perpendicular[first_idx_inside_polygon] - x_around[idx_middle_point], y_perpendicular[first_idx_inside_polygon] - y_around[idx_middle_point])
            dist_last_idx_inside_polygon = np.hypot(x_perpendicular[last_idx_inside_polygon] - x_around[idx_middle_point], y_perpendicular[last_idx_inside_polygon] - y_around[idx_middle_point])
            
            # Distance between the middle point and the other side of the island
            max_distance_inside_polygon = np.max([dist_first_idx_inside_polygon, dist_last_idx_inside_polygon])

            if dist_first_idx_inside_polygon < dist_last_idx_inside_polygon:

                # To avoid the transect to be too close to the other side island, we take a reduced number of points inside the polygon
                if (max_distance_inside_polygon/2) < self.length_transect:
                    effective_length_transect = int(max_distance_inside_polygon/2)

                else:
                    effective_length_transect = self.length_transect

                # Reduce the number of points inside the polygon (island)
                idx_inside_polygon_reduced = idx_inside_polygon[np.argmin(np.abs(np.hypot(x_perpendicular[idx_inside_polygon] - x_around[idx_middle_point], \
                                                                                          y_perpendicular[idx_inside_polygon] - y_around[idx_middle_point]) \
                                                                                         - effective_length_transect))]

                # Number of points outside of the polygon (island)
                idx_outside_polygon_reduced = np.argmin(np.abs(np.hypot(x_perpendicular[:idx_inside_polygon[0]] - x_around[idx_middle_point], \
                                                                                          y_perpendicular[:idx_inside_polygon[0]] - y_around[idx_middle_point]) \
                                                                                         - self.length_transect))

                # Get vectors including the same distance outside of the polygon (island)
                x_transect = x_perpendicular[idx_outside_polygon_reduced:idx_inside_polygon_reduced] 
                y_transect = y_perpendicular[idx_outside_polygon_reduced:idx_inside_polygon_reduced]
                vector_transect = np.column_stack((x_transect, y_transect))

            else:
                # Flip the order of indices inside the polygon
                idx_inside_polygon = np.flip(idx_inside_polygon)

                # To avoid the transect to be too close to the other side island, we take a reduced number of points inside the polygon
                if (max_distance_inside_polygon/2) < self.length_transect:
                    effective_length_transect = int(max_distance_inside_polygon/2)

                else:
                    effective_length_transect = self.length_transect

                # Reduce the number of points inside the polygon (island)
                idx_inside_polygon_reduced = idx_inside_polygon[np.argmin(np.abs(np.hypot(x_perpendicular[idx_inside_polygon] - x_around[idx_middle_point], \
                                                                                          y_perpendicular[idx_inside_polygon] - y_around[idx_middle_point]) \
                                                                                         - effective_length_transect))]

                # Number of points outside of the polygon (island)
                idx_outside_polygon_reduced = np.argmin(np.abs(np.hypot(x_perpendicular[idx_inside_polygon[0]+1:] - x_around[idx_middle_point], \
                                                                                          y_perpendicular[idx_inside_polygon[0]+1:] - y_around[idx_middle_point]) \
                                                                                         - self.length_transect))
                
                # Get vectors including the same distance outside of the polygon (island)
                x_transect = x_perpendicular[idx_inside_polygon_reduced:idx_inside_polygon[0]+1+idx_outside_polygon_reduced] 
                y_transect = y_perpendicular[idx_inside_polygon_reduced:idx_inside_polygon[0]+1+idx_outside_polygon_reduced]
                vector_transect = np.column_stack((x_transect, y_transect))

            # Get the first and last point of the transect
            transect_inside_outside = vector_transect[[0, -1], :]

            # To make sure that the transect is oriented from the island to the ocean (inside -> outside)
            if polygon_shoreline.contains(shapely.geometry.Point(transect_inside_outside[0, 0], transect_inside_outside[0, 1])):
                transects[idx_transect] = transect_inside_outside

            else:
                transects[idx_transect] = np.flip(transect_inside_outside, axis=0)
        
        # Save transects in dictionary
        self.island_info['spatial_reference']['transects'] = transects

        # TODO: improve this
        # Retrieve spatial information about the atoll (if available)
        for value_general_info in self.island_info['general_info'].values():
            if 'atoll' in value_general_info.lower():
                atoll = value_general_info
                break
        
        # If atoll information is available -> OSMNX information
        try:
            geometry_atoll = ox.geocode_to_gdf(atoll)
        
        except:
            geometry_atoll = None

        # Characterise transects (pointing towards the ocean or towards the lagoon)
        if geometry_atoll is not None:
            lat_atoll, lon_atoll = geometry_atoll.lat, geometry_atoll.lon
            dict_transect_direction = {}

            # Loop over the transects
            for transect in transects:
                # Dot product between the transect vector and the vector from the atoll to the middle point of the transect
                dot_product_atoll_transect = np.dot(transects[transect][1, :] - transects[transect][0, :], 
                                                    np.array([lon_atoll-transects[transect][0, 0], 
                                                    lat_atoll-transects[transect][0, 1]]))
                
                if dot_product_atoll_transect > 0:
                    dict_transect_direction[transect] = 'lagoon'
                
                else:
                    dict_transect_direction[transect] = 'ocean or sea'
            
            # Save information in dictionary
            self.island_info['spatial_reference']['transects_direction'] = dict_transect_direction

        # Plot results
        if self.plot_results:
            plt.figure()
            plt.plot(reference_shoreline[:, 0], reference_shoreline[:, 1], 'k')
            for idx_transect in transects.keys():
                plt.plot(transects[idx_transect][:, 0], transects[idx_transect][:, 1], 'r')
                plt.text(transects[idx_transect][-1, 0], transects[idx_transect][-1, 1], idx_transect, fontsize=12)
            plt.axis('equal')
            plt.savefig(os.path.join(self.coastsat_data_path, '_'.join([self.island, self.country]), 'transects.png'), dpi=300)
            plt.show(block=False)

    def get_timeseries(self):
        """
        Retrieve time series data of coastline positions along transects using CoastSat.
        """
        # Define ara of interest
        polygon = [self.island_info['spatial_reference']['polygon'].getInfo()['coordinates'][0]]

        # Define date range
        dates = self.date_range

        # Define site name
        sitename = '_'.join([self.island, self.country])

        # Define path to save data
        filepath_data = self.coastsat_data_path

        # Define list of satellites
        sat_list = self.sat_list

        # Define collection
        collection = self.collection

        # Put all the inputs into a dictionnary
        inputs = {'polygon': polygon,
            'dates': dates,
            'sat_list': sat_list,
            'sitename': sitename,
            'filepath': filepath_data,
            'landsat_collection': collection}
        
        if self.reference_shoreline_transects_only:
            try:
                _ = ox.features_from_place(', '.join([self.island, self.country]), {'natural': 'coastline'})
                metadata = None
            
            except:
                # Check if data is available
                if os.path.exists(os.path.join(filepath_data, sitename)) and not self.overwrite:
                    metadata = SDS_download.get_metadata(inputs)
                
                # If data is not available
                else:
                    inputs['dates'] = ['2021-01-01', '2021-03-01']
                    metadata = SDS_download.retrieve_images(inputs)

        else:
            # Check if data is available
            if os.path.exists(os.path.join(filepath_data, sitename)) and not self.re_download:
                metadata = SDS_download.get_metadata(inputs)
            
            # If data is not available
            else:
                metadata = SDS_download.retrieve_images(inputs)            
        
        # Settings for shoreline extraction
        settings = { 
            # general parameters:
            'cloud_thresh': 0.5,        # threshold on maximum cloud cover
            'dist_clouds': 20,         # distance around clouds where shoreline can't be mapped
            'output_epsg': 3857,       # epsg code of spatial reference system desired for the output

            # quality control:
            'check_detection': False,    # if True, shows each shoreline detection to the user for validation
            'adjust_detection': False,  # if True, allows user to adjust the postion of each shoreline by changing the threshold
            'save_figure': True,        # if True, saves a figure showing the mapped shoreline for each image

            # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
            'min_beach_area': 50,     # minimum area (in metres^2) for an object to be labelled as a beach
            'min_length_sl': 50,       # minimum length (in metres) of shoreline perimeter to be valid
            'cloud_mask_issue': False,  # switch this parameter to True if sand pixels are masked (in black) on many images  
            'sand_color': 'default',    # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
            'pan_off': False,           # True to switch pansharpening off for Landsat 7/8/9 imagery
            'max_dist_ref': 30,         # maximum distance (in pixels) between a valid shoreline and the reference shoreline

            # add the inputs defined previously
            'inputs': inputs,
        }

        if not self.reference_shoreline_transects_only:

            # Save .jpg of the preprocessed RGB images
            if not os.path.exists(os.path.join(filepath_data, sitename, 'jpg_files', 'preprocessed')) and not self.overwrite:
                SDS_preprocess.save_jpg(metadata, settings, use_matplotlib=True)

        # Get reference shoreline
        reference_shoreline = self.get_reference_shoreline(metadata, settings)
        settings['reference_shoreline'] = reference_shoreline

        # Save information in dictionary
        self.island_info['spatial_reference']['reference_shoreline'] = reference_shoreline
        self.island_info['timeseries_{}'.format(self.acronym)]['inputs'] = inputs
        self.island_info['timeseries_{}'.format(self.acronym)]['settings'] = settings

        if not self.reference_shoreline_transects_only:
            # Output file
            file_output = os.path.join(filepath_data, sitename, sitename + '_output.pkl')

            # Check if shoreline positions have already been extracted
            if os.path.exists(file_output) and not self.overwrite and not self.extract_shorelines:
                with open(file_output, 'rb') as f:
                    output = pickle.load(f)

            # Extract shoreline positions
            else:
                output = SDS_shoreline.extract_shorelines(metadata, settings)

            # Removes duplicates (images taken on the same date by the same satellite)
            output = SDS_tools.remove_duplicates(output, label='shorelines_waterline')
            output = SDS_tools.remove_duplicates(output, label='shorelines_vegline')

            # Remove inaccurate georeferencing (set threshold to 10 m)
            output = SDS_tools.remove_inaccurate_georef(output, 10) 

            # Plot mapped shorelines
            if self.plot_results:

                # Define figure
                fig = plt.figure(figsize=[15, 8])

                # Plot every shoreline
                for i in range(len(output['shorelines_0'])):
                    sl = output['shorelines_0'][i]
                    date = output['dates'][i]
                    plt.plot(sl[:, 0], sl[:, 1], '.', label=date.strftime('%d-%m-%Y'))

                # Aesthetic parameters
                plt.legend()
                plt.axis('equal')
                plt.xlabel('Eastings')
                plt.ylabel('Northings')
                plt.grid(linestyle=':', color='0.5')
                plt.show(block=False)
        
        # Get transects
        if 'transects' in self.island_info['spatial_reference'].keys() and not self.overwrite:
            transects = self.island_info['spatial_reference']['transects']
        
        else:
            self.get_transects()
            transects = self.island_info['spatial_reference']['transects']

        if not self.reference_shoreline_transects_only:
            
            # Along-shore distance over which to consider shoreline points to compute the median intersection
            settings_transects = {'along_dist': 25}
            cross_distance_waterline = SDS_transects.compute_intersection(output, transects, settings_transects, label='shorelines_waterline') 
            cross_distance_vegline = SDS_transects.compute_intersection(output, transects, settings_transects, label='shorelines_vegline')

            # Remove outliers
            settings_outliers = {'max_cross_change': 40,             # maximum cross-shore change observable between consecutive timesteps
                                'otsu_threshold': [-1.1, 0.5],        # min and max intensity threshold use for contouring the shoreline
                                'plot_fig': False}           # whether to plot the intermediate steps
                                
            cross_distance_waterline_no_outliers = SDS_transects.reject_outliers(cross_distance_waterline, output, settings_outliers, label='waterline') 
            cross_distance_vegline_no_outliers = SDS_transects.reject_outliers(cross_distance_vegline, output, settings_outliers, label='vegline')       

            # Create a dictionary with results
            dict_timeseries = {'datetime': output['dates']}

            # Loop over transects
            for key in cross_distance_waterline.keys():
                dict_timeseries['coastline_position_transect_{}_waterline'.format(key)] = cross_distance_waterline[key]

            for key in cross_distance_waterline_no_outliers.keys():    
                dict_timeseries['coastline_position_transect_{}_waterline_no_outliers'.format(key)] = cross_distance_waterline_no_outliers[key]

            for key in cross_distance_vegline.keys():
                dict_timeseries['coastline_position_transect_{}_vegline'.format(key)] = cross_distance_vegline[key]
            
            for key in cross_distance_vegline_no_outliers.keys():
                dict_timeseries['coastline_position_transect_{}_vegline_no_outliers'.format(key)] = cross_distance_vegline_no_outliers[key]

            # Create and save DataFrame
            df_timeseries = pd.DataFrame(dict_timeseries).set_index('datetime')
            fn = os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'], 'transect_time_series.csv')
            df_timeseries.to_csv(fn, sep=',')
            
            # Save information in dictionary
            self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = df_timeseries

# --------------------------------------------------------------------------------------------------------------------------------------------------------  
class TimeSeriesERA5(IslandTimeBase):
    """
    Retrieves and calculates time series data of climate/weather variables from remote sensing (ERA5)
    for a specified island.

    Args:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): If True, print initialisation information. Default is True.
        overwrite (bool, optional): If True, overwrite existing data. Default is False.
    """
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.acronym = 'ERA5'
        self.name_description = 'ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate'
        self.description = 'Retrieves and calculates time series data of climate/weather variables from remote sensing (ERA5) for a given island.'
        self.description_timeseries = 'TODO: Available information: winds, temperature, precipitation, evaporation'
        self.source = 'Copernicus Climate Change Service (C3S) (2017): ERA5: Fifth generation of ECMWF atmospheric reanalyses of the global climate. Copernicus Climate Change Service Climate Data Store (CDS), (date of access), https://cds.climate.copernicus.eu/cdsapp#!/home'

    def get_timeseries_winds(self, data_ERA5, transects):       

        # u-component of wind
        timeseries_u10 = data_ERA5['u10'].sel(longitude=self.island_info['spatial_reference']['longitude'], 
                                              latitude=self.island_info['spatial_reference']['latitude'], 
                                              method='nearest')
        
        # v-component of wind
        timeseries_v10 = data_ERA5['v10'].sel(longitude=self.island_info['spatial_reference']['longitude'], 
                                              latitude=self.island_info['spatial_reference']['latitude'], 
                                              method='nearest')
        
        # Take mean of expver
        if 'expver' in list(data_ERA5.variables):
            timeseries_u10 = timeseries_u10.mean(dim='expver')
            timeseries_v10 = timeseries_v10.mean(dim='expver')
        
        # Get wind speed (speed = sqrt(u^2 + v^2))
        timeseries_wind_speed = np.hypot(timeseries_u10, timeseries_v10)

        # Get wind direction (direction = arctan(v/u))
        timeseries_wind_direction = np.degrees(np.arctan2(timeseries_v10, timeseries_u10)) % 360

        # Get wind direction in true degrees
        timeseries_wind_direction_true = (270 - np.degrees(np.arctan2(timeseries_v10, timeseries_u10))) % 360

        # Create DataFrame
        df_ERA5_winds = pd.DataFrame({'wind_speed_10m': timeseries_wind_speed.values,
                                      'wind_direction_10m': timeseries_wind_direction.values,
                                      'wind_direction_true_10m': timeseries_wind_direction_true.values},
                                      index=data_ERA5['time'])
        
        # Create time series specific to each transect -> wind speed in the direction of the transect
        # Create wind vector
        wind_vector = np.array([timeseries_u10.values, timeseries_v10.values])

        if 'timeseries_transect_specific' not in self.island_info['timeseries_{}'.format(self.acronym)].keys():
            self.island_info['timeseries_{}'.format(self.acronym)]['timeseries_transect_specific'] = {}

        # Loop in every transect
        for transect in transects.keys():
            # Transect vector
            transect_vector = np.array([transects[transect][1, 0]-transects[transect][0, 0], 
                                        transects[transect][1, 1]-transects[transect][0, 1]])
            
            # Unit transect vector
            unit_transect_vector = transect_vector / np.linalg.norm(transect_vector)

            # Dot product
            wind_speed_parallel_transect = np.dot(unit_transect_vector, wind_vector)

            self.island_info['timeseries_{}'.format(self.acronym)]['timeseries_transect_specific'][transect] = pd.DataFrame({'wind_speed_parallel_transect': wind_speed_parallel_transect}, index=data_ERA5['time'])

        return df_ERA5_winds

    def get_timeseries_wave_energy(self, data_ERA5, significant_wave_height, mean_wave_period, name):

        # Select variables
        wave_height = data_ERA5[significant_wave_height].sel(longitude=self.island_info['spatial_reference']['longitude'], 
                                                     latitude=self.island_info['spatial_reference']['latitude'], 
                                                     method='nearest')
        wave_period = data_ERA5[mean_wave_period].sel(longitude=self.island_info['spatial_reference']['longitude'],
                                                        latitude=self.island_info['spatial_reference']['latitude'],
                                                        method='nearest')
        # Take mean of expver
        if 'expver' in list(data_ERA5.variables):
            wave_height = wave_height.mean(dim='expver')
            wave_period = wave_period.mean(dim='expver')        

        # SI units
        rho =  1025 # kg/m^3
        g = 9.81 # m/s^2

        # Wave energy flux in watts per meter of wave front
        wave_energy = (rho * (g ** 2) * (wave_height ** 2) * wave_period) / (64 * np.pi)

        # Convert to kW/m
        wave_energy = wave_energy / 1000.

        # Create DataFrame
        df_ERA5_wave_energy = pd.DataFrame({'wave_energy_of_{}'.format(name): wave_energy}, index=data_ERA5['time'])

        return df_ERA5_wave_energy

    def get_timeseries(self): 

        print('~ Retrieving time series. ~')

        # Define area of interest (whole country)
        if 'area_country' in self.island_info['spatial_reference'].keys():
            area_country = self.island_info['spatial_reference']['area_country']
        
        else:
            geometry_ox_country = ox.geocode_to_gdf(self.country)
            area_country = [geometry_ox_country.bbox_north.values[0], \
                            geometry_ox_country.bbox_west.values[0], \
                            geometry_ox_country.bbox_south.values[0], \
                            geometry_ox_country.bbox_east.values[0]]
            self.island_info['spatial_reference']['area_country'] = area_country

        # Retrieve transects
        if 'transects' in self.island_info['spatial_reference'].keys():
            transects = self.island_info['spatial_reference']['transects']

        else:
            _, transects = TimeSeriesCoastSat(self.island, self.country, verbose_init=False, reference_shoreline_transects_only=True).main()

        # Define cdsapi key
        cds = cdsapi.Client(url="https://cds.climate.copernicus.eu/api/v2", key="200721:d13b27b3-32f8-4315-a9c0-e65dc3eb6fdd")

        # Query cdsapi request
        fl = cds.retrieve(
            'reanalysis-era5-single-levels-monthly-means',
            {
                'format': 'netcdf',
                'product_type': 'monthly_averaged_reanalysis',
                'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind',
                             '2m_dewpoint_temperature', '2m_temperature', 
                             'soil_temperature_level_1', 'total_precipitation',
                             'evaporation', 'sea_surface_temperature', 'mean_sea_level_pressure',
                ],
                'time': '00:00',
                'year': [
                    '2010', '2011', '2012', 
                    '2013', '2014', '2015',
                    '2016', '2017', '2018',
                    '2019', '2020', '2021',
                    '2022',
                ],
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'area': area_country,
            })

        # Query cdsapi request
        fl_waves = cds.retrieve(
            'reanalysis-era5-single-levels-monthly-means',
            {
                'format': 'netcdf',
                'product_type': 'monthly_averaged_reanalysis',
                'variable': ['mean_direction_of_total_swell', 'mean_direction_of_wind_waves', 
                             'mean_period_of_total_swell', 'mean_period_of_wind_waves', 
                             'mean_wave_direction', 'mean_wave_period',
                             'significant_height_of_combined_wind_waves_and_swell',
                             'significant_height_of_total_swell', 'significant_height_of_wind_waves',
                ],
                'time': '00:00',
                'year': [
                    '2010', '2011', '2012', 
                    '2013', '2014', '2015',
                    '2016', '2017', '2018',
                    '2019', '2020', '2021',
                    '2022',
                ],
                'month': [
                    '01', '02', '03',
                    '04', '05', '06',
                    '07', '08', '09',
                    '10', '11', '12',
                ],
                'area': area_country,
            })
        
        # Open file as xarray dataset
        with urllib.request.urlopen(fl.location) as f:
            data_ERA5 = xr.open_dataset(f.read())
        
        with urllib.request.urlopen(fl_waves.location) as f:
            data_ERA5_waves = xr.open_dataset(f.read())

        # Winds
        df_ERA5_winds = self.get_timeseries_winds(data_ERA5, transects)

        # Wave energy (wind waves, swell and combined)
        df_ERA5_wave_energy_swell = self.get_timeseries_wave_energy(data_ERA5_waves, 'shts', 'mpts', 'total_swell')
        df_ERA5_wave_energy_wind_waves = self.get_timeseries_wave_energy(data_ERA5_waves, 'shww', 'mpww', 'wind_waves')
        df_ERA5_wave_energy_combined = self.get_timeseries_wave_energy(data_ERA5_waves, 'swh', 'mwp', 'combined_wind_waves_and_swell')

        # Create DataFrame and units dictionary
        df_ERA5 = pd.DataFrame(index=data_ERA5['time'])
        units_ERA5 = {'wind_speed_10m': 'm/s',
                      'wind_direction_10m': 'degree',
                      'wind_direction_true_10m': 'degree true',
                      'wave_energy_total_swell': 'kW/m',
                      'wave_energy_wind_waves': 'kW/m',
                      'wave_energy_combined_wind_waves_and_swell': 'kW/m'}

        # Loop in every remaining variable
        for var_ERA5 in (list(data_ERA5.variables) + list(data_ERA5_waves.variables)):
            if var_ERA5 in ['longitude', 'latitude', 'time', 'expver', 'month_number', 'year', 'u10', 'v10', 'shts', 'mpts', 'shww', 'mpww', 'swh', 'mwp']:
                continue    

            if var_ERA5 in list(data_ERA5.variables):
                data_to_use = data_ERA5
            
            else:
                data_to_use = data_ERA5_waves

            # Get long name and units
            long_name_var = data_to_use[var_ERA5].attrs['long_name'].replace(' ', '_').replace('-', '_').lower()
            unit = data_to_use[var_ERA5].attrs['units']

            # Get time series
            timeseries_var = data_to_use[var_ERA5].sel(longitude=self.island_info['spatial_reference']['longitude'], 
                                                     latitude=self.island_info['spatial_reference']['latitude'], 
                                                     method='nearest')

            # Take mean of expver (if needed)
            if 'expver' in list(data_to_use.variables):
                timeseries_var = timeseries_var.mean(dim='expver')

            df_ERA5[long_name_var] = timeseries_var
            units_ERA5[long_name_var] = unit

        # Combine DataFrames
        df_ERA5_total = pd.concat([df_ERA5, df_ERA5_winds, df_ERA5_wave_energy_swell, 
                                   df_ERA5_wave_energy_wind_waves, df_ERA5_wave_energy_combined], axis=1)
        
        # Save information in dictionary
        df_ERA5_total = df_ERA5_total.apply(pd.to_numeric)
        self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = df_ERA5_total
        self.island_info['timeseries_{}'.format(self.acronym)]['units'] = units_ERA5

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class AddEcologicalCoastalUnits(IslandTimeBase):
    """
    A class for retrieving Ecological Coastal Units (ECUs) and adding the information for each transect.

    Attributes:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): Whether to display verbose initialisation messages. Default is True.
        overwrite (bool, optional): Whether to overwrite existing data. Default is False.
    """
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.acronym = 'ECU'
        self.name_description = 'Ecological Coastal Units (ECUs)'
        self.description = 'This module retrieves Ecological Coastal Units (ECUs) and add the information for each transect.'
        self.source = 'https://www.esri.com/arcgis-blog/products/arcgis-living-atlas/mapping/ecus-available/'
        self.ECU_path = os.path.join(os.getcwd(), 'data', 'ECU')
        self.dict_rename = {'MEAN_SIG_W': 'Mean Significant Wave Height', 
                            'TIDAL_RANG': 'Tidal Range', 
                            'CHLOROPHYL': 'Chlorophyll-a', 
                            'TURBIDITY': 'Turbidity',
                            'TEMP_MOIST': 'Temperature Moist',
                            'EMU_PHYSIC': 'EMU Physical',
                            'REGIONAL_S': 'Regional Sinuosity',
                            'MAX_SLOPE': 'Max Slope',
                            'OUTFLOW_DE': 'Outflow Density',
                            'ERODIBILIT': 'Erodibility',
                            'LENGTH_GEO': 'Geographical Length',
                            'chl_label': 'Chlorophyll-a Descriptor',
                            'river_labe': 'River Descriptor',
                            'sinuosity_': 'Sinuosity Descriptor',
                            'slope_labe': 'Slope Descriptor',
                            'tidal_labe': 'Tidal Descriptor',
                            'turbid_lab': 'Turbidity Descriptor',
                            'wave_label': 'Wave Descriptor',
                            'CSU_Descri': 'CSU Descriptor',
                            'CSU_ID': 'CSU ID',
                            'OUTFLOW__1': 'Outflow Density Rescaled',
                            'Shape_Leng': 'Shape Length',
                            'geometry': 'Geometry'}

    def add_info(self):
        """
        Retrieve Ecological Coastal Units (ECUs) and add the information for each transect.
        """
        print('~ Retrieving Ecological Coastal Units. ~')
        
        # Retrieve transects and reference shoreline
        if 'transects' in self.island_info['spatial_reference'].keys():
            transects = self.island_info['spatial_reference']['transects']
            reference_shoreline = self.island_info['spatial_reference']['reference_shoreline']
        else:
            reference_shoreline, transects = TimeSeriesCoastSat(self.island, self.country, verbose_init=False, reference_shoreline_transects_only=True).main()

        # Read ECU shapefile
        shapefile_ECU = os.path.join(self.ECU_path, 'ECU_{}_shapefile.shp'.format(self.country))
        gdf_ECU = gpd.read_file(shapefile_ECU)
        geometry_ECU = gdf_ECU.geometry

        # Create empty dictionary for transect ECU characteristics
        transects_ECU_characteristics = {}

        for key in transects.keys():
            transect = transects[key]
            linestring_transect = shapely.geometry.LineString(transect)

            # Retrieve ECU for this transect
            try:
                # Index of ECU that intersects with transect
                idx_transect_ECU = np.argwhere(np.array([linestring_transect.intersects(linestring_ECU) for linestring_ECU in geometry_ECU])).flatten()[0]

                # Get corresponding ECU
                df_transect_ECU = gdf_ECU[gdf_ECU.index==idx_transect_ECU].set_index('MasterKey')

            # ECU not available for this transect    
            except:
                continue

            # Rename columns for aesthetics
            df_transect_ECU = df_transect_ECU.rename(columns=self.dict_rename)

            # Save information in dictionary
            transects_ECU_characteristics[key] = df_transect_ECU
        
        # Save information in dictionary
        self.island_info['spatial_reference']['reference_shoreline'] = reference_shoreline
        self.island_info['spatial_reference']['transects'] = transects
        self.island_info['characteristics_{}'.format(self.acronym)]['transects_characteristics_{}'.format(self.acronym)] = transects_ECU_characteristics

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TimeSeriesVegetation(IslandTimeBase):
    """
    A class for calculating time series of vegetation health (NDVI) from remote sensing for a given island.

    Attributes:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): Whether to display verbose initialisation messages. Default is True.
        overwrite (bool, optional): Whether to overwrite existing data. Default is False.
    """
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.info_satellite_path = os.path.join(os.getcwd(), 'data', 'info_satellite')
        self.coastsat_data_path = os.path.join(os.getcwd(), 'data', 'coastsat_data', self.island + '_' + self.country)
        self.thresholds_NDVI = [0.25, 0.90]
        self.acronym = 'vegetation'
        self.name_description = 'Vegetation health (NDVI)'
        self.description = 'This module calculates time series of vegetation health (NDVI) from remote sensing for a given island.'
        self.description_timeseries = 'Time series of the average NDVI value across the island.'
        self.source = 'This work'

    def create_vegetation_masks(self, sat, metadata_sat):
        """
        Create vegetation masks for a specific satellite.

        Args:
            sat (str): The satellite name.
            metadata_sat (dict): Metadata for the satellite.

        Returns:
            numpy.ndarray, numpy.ndarray: The total vegetation mask and coastal vegetation mask.
        """
        if all(element in self.island_info['timeseries_{}'.format(self.acronym)].keys() for element in \
                                          ['mask_total_vegetation_{}'.format(sat), 'mask_coastal_vegetation_{}'.format(sat), \
                                           'mask_transects_vegetation_{}'.format(sat)]):
            print('~ Vegetation masks already available for this satellite. Returning data. ~')
            return self.island_info['timeseries_{}'.format(self.acronym)]['mask_total_vegetation_{}'.format(sat)], self.island_info['timeseries_{}'.format(self.acronym)]['mask_coastal_vegetation_{}'.format(sat)] #, self.island_info['timeseries_{}'.format(self.acronym)]['mask_transects_vegetation_{}'.format(sat)]

        print('~ Creating vegetation masks. ~')
        
        # Retrieve transects, metadata and reference shoreline
        if 'transects' in self.island_info['spatial_reference'].keys():
            transects = self.island_info['spatial_reference']['transects']
            reference_shoreline = self.island_info['spatial_reference']['reference_shoreline']
        
        else:
            reference_shoreline, transects = TimeSeriesCoastSat(self.island, self.country, verbose_init=False, reference_shoreline_transects_only=True).main()
            self.island_info['spatial_reference']['reference_shoreline'] = reference_shoreline
            self.island_info['spatial_reference']['transects'] = transects 

        # Define EPSG code of the images
        list_epsg = np.unique(np.array(metadata_sat['epsg']), return_counts=True)
        image_epsg = list_epsg[0][np.argmax(list_epsg[1])]

        # Open any multispectral image to retrieve spatial information
        list_files_ms = os.listdir(os.path.join(self.coastsat_data_path, sat, 'ms'))
        file_ms = gdal.Open(os.path.join(self.coastsat_data_path, sat, 'ms', list_files_ms[-1]), gdal.GA_ReadOnly)

        # Extract geotransform parameters
        geotransform = file_ms.GetGeoTransform()
        x_origin = geotransform[0]
        y_origin = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]

        # Get the dimensions of the raster
        width = file_ms.RasterXSize
        height = file_ms.RasterYSize

        # Create a mesh grid of coordinates
        x_img = np.arange(x_origin, x_origin + (width * pixel_width), pixel_width)
        y_img = np.arange(y_origin, y_origin + (height * pixel_height), pixel_height)

        # Use numpy.meshgrid to create the mesh grid
        xx_img, yy_img = np.meshgrid(x_img, y_img)

        # Reproject transects and reference shoreline
        src_crs = pyproj.CRS('EPSG:3857')
        tgt_crs = pyproj.CRS('EPSG:{}'.format(image_epsg))
        transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

        # Reproject reference shoreline
        x_reprojected_shoreline, y_reprojected_shoreline = transformer.transform(reference_shoreline[:, 0], reference_shoreline[:, 1])
        reference_shoreline_reprojected = np.column_stack((x_reprojected_shoreline, y_reprojected_shoreline))

        # Reproject transects
        '''
        for transect in transects.keys():
            x_reprojected_transect, y_reprojected_transect = transformer.transform(transects[transect][:, 0], transects[transect][:, 1])
            transects[transect] = np.column_stack((x_reprojected_transect, y_reprojected_transect))
        '''

        # Create shapely.geometry.Polygon and shapely.geometry.LineString objects
        polygon_reference_shoreline = shapely.geometry.Polygon(reference_shoreline_reprojected)
        linestring_reference_shoreline = shapely.geometry.LineString(reference_shoreline_reprojected)

        # Create an empty mask array with the same shape as the meshgrid
        mask_total_vegetation, mask_coastal_vegetation = np.zeros(xx_img.shape, dtype=bool), np.zeros(xx_img.shape, dtype=bool)
        mask_shoreline_buffer = np.zeros(xx_img.shape, dtype=bool)

        # Create dictionary for transect masks
        '''
        mask_transects_vegetation = {}
        for transect in transects.keys():
            mask_transects_vegetation[transect] = np.zeros(xx_img.shape, dtype=bool)
        '''

        # Fill masks
        for i in tqdm(range(xx_img.shape[0])):
            for j in range(xx_img.shape[1]):
                point = shapely.geometry.Point(xx_img[i, j], yy_img[i, j])

                # Shoreline buffer
                if point.distance(linestring_reference_shoreline) < 75:
                    mask_shoreline_buffer[i, j] = True

                # Total vegetation
                if polygon_reference_shoreline.contains(point):
                    mask_total_vegetation[i, j] = True
                                
                # Coastal vegetation
                if polygon_reference_shoreline.contains(point) and point.distance(linestring_reference_shoreline) < 100:
                    mask_coastal_vegetation[i, j] = True
                
                # Transect vegetation
                '''
                for transect in transects.keys():
                    linestring_transect = shapely.geometry.LineString(transects[transect])
                    if polygon_reference_shoreline.contains(point) and point.distance(linestring_reference_shoreline) < 100 and point.distance(linestring_transect) < 100:
                        mask_transects_vegetation[transect][i, j] = True
                '''

        # Save masks in dictionary
        self.island_info['timeseries_{}'.format(self.acronym)]['mask_total_vegetation_{}'.format(sat)] = mask_total_vegetation
        self.island_info['timeseries_{}'.format(self.acronym)]['mask_coastal_vegetation_{}'.format(sat)] = mask_coastal_vegetation
        self.island_info['spatial_reference']['reference_shoreline_buffer_{}'.format(sat)] = mask_shoreline_buffer
        #self.island_info['timeseries_{}'.format(self.acronym)]['mask_transects_vegetation_{}'.format(sat)] = mask_transects_vegetation

        return mask_total_vegetation, mask_coastal_vegetation #, mask_transects_vegetation
    
    def get_timeseries(self):
        """
        Calculate the time series of vegetation health (NDVI) for the island.
        """       
        # Open metadata file
        metadata = pickle.load(open(os.path.join(self.coastsat_data_path, '{}_{}_metadata.pkl'.format(self.island, self.country)), 'rb'))

        # Retrieve settings
        settings = self.island_info['timeseries_coastsat']['settings']
        
        # Iteration variable
        idx_sat = 0

        # Loop in every satellite
        for sat in metadata.keys():
            print(sat)

            # Empty metadata for this satellite
            if metadata[sat]['filenames'] == []:
                continue

            # Retrieve masks
            mask_total_vegetation, mask_coastal_vegetation = self.create_vegetation_masks(sat, metadata[sat]) # mask_transects_vegetation

            # File path and names from CoastSat folder
            filepath = SDS_tools.get_filepath(settings['inputs'], sat)
            filenames = metadata[sat]['filenames']

            # Create empty lists for the time series
            list_datetime = []
            list_NDVI_total_vegetation, list_NDVI_coastal_vegetation = [], []
            #list_NDVI_transects_vegetation = {transect: [] for transect in mask_transects_vegetation.keys()}

            # Loop through images within the folder (taken from SDS_classify.py)
            for idx_img in range(len(filenames)):
                # Get file name
                fn = SDS_tools.get_filenames(filenames[idx_img], filepath, sat)

                # Retrieve information about image
                im_ms, _, cloud_mask, _, _, im_nodata = SDS_preprocess.preprocess_single(fn, sat, settings['cloud_mask_issue'], settings['pan_off'], 'C02')

                # Compute cloud_cover percentage (with no data pixels)
                cloud_cover_combined = np.divide(sum(sum(cloud_mask.astype(int))), (cloud_mask.shape[0] * cloud_mask.shape[1]))
                
                # If 99% of cloudy pixels in image -> skip
                if cloud_cover_combined > 0.99: 
                    continue

                # Remove no data pixels from the cloud mask (for example L7 bands of no data should not be accounted for)
                cloud_mask_adv = np.logical_xor(cloud_mask, im_nodata)

                # Compute updated cloud cover percentage (without no data pixels)
                cloud_cover = np.divide(sum(sum(cloud_mask_adv.astype(int))), (sum(sum((~im_nodata).astype(int)))))

                # Skip image if cloud cover is above threshold
                if cloud_cover > settings['cloud_thresh'] or cloud_cover == 1:
                    continue

                # Get NDVI image
                img_NDVI = SDS_tools.nd_index(im_ms[:, :, 3], im_ms[:, :, 2], cloud_mask)

                try:
                    # Total vegetation
                    img_NDVI_masked_total = np.multiply(img_NDVI, mask_total_vegetation)
                    list_NDVI_total_vegetation.append(np.mean(img_NDVI_masked_total[(img_NDVI_masked_total > self.thresholds_NDVI[0]) & \
                                                                                    (img_NDVI_masked_total < self.thresholds_NDVI[1])]))

                    # Coastal vegetation
                    img_NDVI_masked_coastal = np.multiply(img_NDVI, mask_coastal_vegetation)
                    list_NDVI_coastal_vegetation.append(np.mean(img_NDVI_masked_coastal[(img_NDVI_masked_coastal > self.thresholds_NDVI[0]) & \
                                                                                        (img_NDVI_masked_coastal < self.thresholds_NDVI[1])]))
                    
                    # Transect vegetation
                    '''
                    for transect in mask_transects_vegetation.keys():
                        img_NDVI_masked_transect = np.multiply(img_NDVI, mask_transects_vegetation[transect])
                        list_NDVI_transects_vegetation[transect].append(np.mean(img_NDVI_masked_transect[(img_NDVI_masked_transect > self.thresholds_NDVI[0]) & (img_NDVI_masked_transect < self.thresholds_NDVI[1])]))
                    '''
                                                                                                                                                                                              
                    # Datetime
                    date = fn[0].split('\\')[-1].split('_')[0].split('-')[:3]
                    list_datetime.append(datetime.datetime(year=int(date[0]), month=int(date[1]), day=int(date[2])))

                except:
                    continue
                
            # Create a pd.DataFrame with results
            df_timeseries_sat = pd.DataFrame({'datetime': list_datetime, 
                            'NDVI_total_vegetation_{}'.format(sat): list_NDVI_total_vegetation, 
                            'NDVI_coastal_vegetation_{}'.format(sat): list_NDVI_coastal_vegetation}).set_index('datetime')
            
            '''
            for transect in mask_transects_vegetation.keys():
                df_timeseries_sat['NDVI_transect_{}_{}'.format(transect, sat)] = list_NDVI_transects_vegetation[transect]
            '''

            if idx_sat == 0:
                df_timeseries = df_timeseries_sat
            
            else:
                df_timeseries_sat = df_timeseries_sat.groupby(df_timeseries_sat.index).mean()
                df_timeseries = df_timeseries.groupby(df_timeseries.index).mean()
                df_timeseries = pd.concat([df_timeseries, df_timeseries_sat], axis=1)

            idx_sat += 1

        # Save information in dictionary
        self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = df_timeseries

# --------------------------------------------------------------------------------------------------------------------------------------------------------
class TimeSeriesClimateIndices(IslandTimeBase):
    def __init__(self, island, country, verbose_init=True, overwrite=False):
        super().__init__(island, country, verbose_init, overwrite)
        self.acronym = 'climate_indices'
        self.name_description = 'Global Climate Indices'
        self.description = 'This module retrieves climate indices: monthly atmospheric and ocean time series (e.g., ENSO).'
        self.description_timeseries = {'dmi': 'Dipole Mode Index (Indian Ocean Dipole Index)',
                                       'pna': 'Pacific North American Index', \
                                       'epo': 'East Pacific/North Pacific Oscillation', \
                                       'wp': 'Western Pacific Index', \
                                       'ea': 'Eastern Atlantic/Western Russia', \
                                       'nao': 'North Atlantic Oscillation', \
                                       'jonesnao': 'North Atlantic Oscillation (Jones)', \
                                       'soi': 'Southern Oscillation Index', \
                                       'nina3.anom': 'Eastern Tropical Pacific SST (anomalies)', \
                                       'nina3': 'Eastern Tropical Pacific SST (mean values)', \
                                       'censo': 'Bivariate ENSO Timeseries', \
                                       'tna': 'Tropical Northern Atlantic Index', \
                                       'tsa': 'Tropical Southern Atlantic Index', \
                                       'whwp': 'Western Hemisphere Warm Pool', \
                                       'oni': 'Oceanic Niño Index', \
                                       'meiv2': 'Multivariate ENSO Index (MEI V2)', \
                                       'nina1.anom': 'Extreme Eastern Tropical Pacific SST (anomalies)', \
                                       'nina1': 'Extreme Eastern Tropical Pacific SST (mean values)', \
                                       'nina4.anom': 'Central Tropical Pacific SST (anomalies)', \
                                       'nina4': 'Central Tropical Pacific SST (mean values)', \
                                       'nina34.anom': 'East Central Tropical Pacific SST (anomalies)', \
                                       'nina34': 'East Central Tropical Pacific SST (mean values)', \
                                       'pdo': 'Pacific Decadal Oscillation', \
                                       'ipotpi.hadisst2': 'Tripole Index for the Interdecadal Pacific Oscillation', \
                                       'noi': 'Northern Oscillation Index', \
                                       'np': 'North Pacific Pattern', \
                                       'tni': 'Indices of El Niño Evolution', \
                                       'hurr': 'Monthly Totals Atlantic Hurricanes and Named Tropical Storms', \
                                       'ao': 'Arctic Oscillation', \
                                       'aao': 'Antarctic Oscillation', \
                                       'pacwarm': 'Pacific Warmpool Area Average', \
                                       'eofpac': 'Tropical Pacific SST EOF', \
                                       'atltri': 'Atlantic Tripole SST EOF', \
                                       'amon.us': 'Atlantic multidecadal Oscillation (unsmoothed)', \
                                       'ammsst': 'Atlantic Merodional Mode', \
                                       'NTA_ersst': 'North Tropical Atlantic SST Index', \
                                       'CAR_ersst': 'Caribbean SST Index', \
                                       'amon.sm': 'Atlantic Multidecadal Oscillation (smoothed)', \
                                       'qbo': 'Quasi-Biennial Oscillation', \
                                       'espi': 'ENSO Precipitation Index', \
                                       'indiamon': 'Central Indian Precipitation', \
                                       'sahelrain': 'Sahel Standardized Rainfall', \
                                       'swmonsoon': 'Area Averaged Precipitation for Arizona and New Mexico', \
                                       'brazilrain': 'Northeast Brazil Rainfall Anomaly', \
                                       'solar': 'Solar Flux (10.7cm)', \
                                       'gmsst': 'Global Mean Land/Ocean Temperature', \
                                       }  
        self.source = 'https://psl.noaa.gov/data/climateindices/list/'

    def retrieve_list_urls(self):

        # URL for list of all climate indices
        url_PSL = 'https://psl.noaa.gov/data/climateindices/list/'
        web_PSL = urllib.request.urlopen(url_PSL)
        data_PSL = str(web_PSL.read())

        # Find all URLs within the data from the website
        pattern_url = r'href="([^"]+)"'
        complete_list_urls = np.array(re.findall(pattern_url, data_PSL))

        # Clean list of URLs (only keep data URLs)
        list_urls_first_cleaning = complete_list_urls[np.argwhere((np.char.startswith(complete_list_urls, r'/enso/')) | (np.char.startswith(complete_list_urls, r'/data/'))).T[0]]
        list_urls_second_cleaning = list_urls_first_cleaning[np.argwhere((~np.char.endswith(list_urls_first_cleaning, r'html')) & \
                                                                         (~np.char.endswith(list_urls_first_cleaning, r'/')) & \
                                                                         (np.char.find(list_urls_first_cleaning, "list") == -1) & \
                                                                         (np.char.find(list_urls_first_cleaning, ".long.") == -1)).T[0]][1:]

        # Delete depecrated climate indices (no recent updates)
        list_urls_third_cleaning = np.delete(list_urls_second_cleaning, np.argwhere(list_urls_second_cleaning=='/data/correlation/trend.data')[0][0])
        list_urls_cleaned = np.delete(list_urls_third_cleaning, np.argwhere(list_urls_third_cleaning=='/data/correlation/glaam.data.scaled')[0][0]) 
        
        # Save information
        self.list_urls = list_urls_cleaned

    def get_timeseries(self):

        # Retrieve list of URL from NOAA PSL website
        self.retrieve_list_urls()

        # Loop through all climate indices
        for idx, climate_index in enumerate(tqdm(self.island_info['timeseries_{}'.format(self.acronym)]['description_timeseries'].keys())):

            #print('~ Retrieving {}. ~'.format(self.island_info['timeseries_{}'.format(self.acronym)]['description_timeseries'][climate_index]))

            if climate_index == 'dmi':
                url_climate_index = 'https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data'
                # Retrieve data from URL
                web_climate_index = urllib.request.urlopen(url_climate_index)
                data_climate_index = str(web_climate_index.read())   

            else:
                # Find corresponding URL
                url_climate_index = self.list_urls[np.argwhere(np.char.endswith(self.list_urls, '{}.data'.format(climate_index)))[0][0]]

                try:
                    # Retrieve data from URL
                    web_climate_index = urllib.request.urlopen('https://psl.noaa.gov' + url_climate_index)
                    data_climate_index = str(web_climate_index.read())

                except:
                    print('Climate index not available.')
                    continue

            # Split into a list and retrieve metadata
            list_data_climate_index = data_climate_index.replace('\\t', '').split('\\n')
            list_data_climate_index = [row.strip() for row in list_data_climate_index]
            _, startyear, endyear = list_data_climate_index[0].replace("'", ' ').split()
            nan_value = list_data_climate_index[np.argwhere(np.char.startswith(list_data_climate_index, endyear))[0][0] + 1].strip()
            
            # Keep actual data
            actual_data_climate_index = list_data_climate_index[np.argwhere(np.char.startswith(list_data_climate_index, startyear))[0][0] : \
                                                                np.argwhere(np.char.startswith(list_data_climate_index, endyear))[0][0] + 1]

            # Loop through every row -> data cleaning & stacking
            for row in range(len(actual_data_climate_index)):

                # String -> array
                row_cleaned = np.fromstring(actual_data_climate_index[row], dtype=float, sep=' ')

                # Store data
                if row == 0: 
                    arr_data_climate_index = row_cleaned
                else: 
                    arr_data_climate_index = np.row_stack((arr_data_climate_index, row_cleaned))

            # Array -> DataFrame
            df_data_climate_index = pd.DataFrame(arr_data_climate_index, columns=['year', 'janval', 'febval', 'marval', 'aprval', 'mayval', 'junval', 'julval', 'augval', 'sepval', 'octval', 'novval', 'decval'])

            # From this DataFrame -> create new DataFrame to match the format of other time series
            arr_timeseries_climate_index = np.array(['datetime', climate_index], dtype=object)

            # Retrieve data for every year and month
            for idx_year in df_data_climate_index.index:
                for idx_month in range(1, len(df_data_climate_index.columns[1:]) + 1):
                    arr_timeseries_climate_index = np.row_stack((arr_timeseries_climate_index, \
                                                                 np.array([datetime.datetime(year=int(df_data_climate_index.year[idx_year]), \
                                                                                             month=idx_month, day=1), \
                                                                                             df_data_climate_index[df_data_climate_index.columns[1:][idx_month-1]][idx_year]])))

            # Replace extreme values by NaN
            arr_timeseries_climate_index[:, 1][arr_timeseries_climate_index[:, 1] == float(nan_value)] = np.nan

            # Build final DataFrame
            df_timeseries_climate_index = pd.DataFrame(arr_timeseries_climate_index[1:], columns=arr_timeseries_climate_index[0])
            df_timeseries_climate_index = df_timeseries_climate_index.set_index('datetime')
            df_timeseries_climate_index = df_timeseries_climate_index.rename(columns=self.description_timeseries)

            # Combine data with other climate indices
            if idx == 0:
                df_timeseries_climate_indices = df_timeseries_climate_index
            
            else:
                df_timeseries_climate_indices = pd.concat([df_timeseries_climate_indices, df_timeseries_climate_index], axis=1)

        # Save information in dictionary
        df_timeseries_climate_indices = df_timeseries_climate_indices.apply(pd.to_numeric)
        self.island_info['timeseries_{}'.format(self.acronym)]['timeseries'] = df_timeseries_climate_indices

# --------------------------------------------------------------------------------------------------------------------------------------------------------
def retrieve_island_info(island, country, island_info_path=os.path.join(os.getcwd(), 'data', 'info_islands'), run_pre_timeseries_steps=True, verbose=True):
    """
    Retrieve information about an island and its timeseries data.

    Args:
        island (str): The name of the island.
        country (str): The name of the country.
        island_info_path (str, optional): The path where island information is stored. Default is 'data/info_islands'.
        run_pre_timeseries_steps (bool, optional): Whether to run preprocessing steps before retrieving information. Default is True.
        verbose (bool, optional): Whether to display verbose messages. Default is True.

    Returns:
        dict: A dictionary containing information about the island.
    """
    if verbose: 
        print('\n-------------------------------------------------------------------')
        print('Retrieving all information available for the island.')
        print('Island:', ', '.join([island, country]))
        print('-------------------------------------------------------------------\n')

    # If the path in which the data will be stored doesn't exist, we create it
    if not os.path.exists(island_info_path): 
        os.makedirs(island_info_path)

    # Check what information is already available
    file_island_info = os.path.join(island_info_path, 'info_{}_{}.data'.format(island, country))

    if os.path.isfile(file_island_info):
        # Load the .data file with pickle
        with open(file_island_info, 'rb') as f:
            island_info = pd.read_pickle(f)

        if verbose:
            print('~ The following information is available: ~\n')
            for info in island_info.keys(): 
                print(info)
                if type(island_info[info]) == dict:
                    for info_sd in island_info[info].keys(): 
                        print('              ', info_sd)
    
    # No file exists
    else:
        if run_pre_timeseries_steps:
            if verbose: 
                print('~ No file exists. Will run `pre_timeseries_steps.py`. ~\n')

            island_info = PreTimeSeries(island, country).main()

        else:
            if verbose: 
                print('~ No file exists. ~\n')
    
    return island_info

def run_all(island, country, verbose_init=True, overwrite=False):
    """
    Run all data retrieval and processing steps for an island.

    Args:
        island (str): The name of the island.
        country (str): The name of the country.
        verbose_init (bool, optional): Whether to display verbose initialisation messages. Default is True.
        overwrite (bool, optional): Whether to overwrite existing data. Default is False.

    Returns:
        dict: A dictionary containing information about the island.
    """
    PreTimeSeries(island, country).main()

    for timeseries_class in IslandTimeBase.__subclasses__():
        timeseries_class(island, country, verbose_init=False, overwrite=overwrite).main()

    # Retrieve the dictionary with currently available information about the island
    island_info = retrieve_island_info(island, country, verbose=verbose_init)

    return island_info

def save_new_island_info(island_info):

    island_info_path = os.path.join(os.getcwd(), 'data', 'info_islands')
    island = island_info['general_info']['island']
    country = island_info['general_info']['country']

    # Save dictionary
    with open(os.path.join(island_info_path, 'info_{}_{}.data'.format(island, country)), 'wb') as f:
        pickle.dump(island_info, f)

    return island_info

def plot_shoreline_transects(island_info):
    # Get reference shoreline and transects
    reference_shoreline = island_info['spatial_reference']['reference_shoreline']
    transects = island_info['spatial_reference']['transects']

    # Plot reference shoreline
    plt.figure()
    plt.plot(reference_shoreline[:, 0], reference_shoreline[:, 1], 'k-')
    
    # Plot transects
    for t in transects:
        plt.plot(transects[t][:, 0], transects[t][:, 1], 'r-')
        plt.text(transects[t][-1, 0], transects[t][-1, 1], t, fontsize=12)

    plt.axis('equal')
    plt.show()