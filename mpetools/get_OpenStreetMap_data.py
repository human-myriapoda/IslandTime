"""
This module allows us to retrieve data from OpenStreetMap (OSM).
FOR NOW: only retrieves roads and buildings and creates a mask for remote sensing data.
TODO: verbose, plot
TODO: add more OSM data.
TODO: better georeferencing.
TODO: deal with bigger islands

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpetools import get_info_islands
import geemap
import os
from tifffile import tifffile
from shapely.geometry import Point
import shapely
from tqdm import tqdm

def queryOpenStreetMap(island_info, info_OSM: list):

    # Specify the name that is used to seach for the data
    place_name = island_info['general_info']['island'] + ', ' + island_info['general_info']['country']

    # Loop over information to retrieve (e.g., roads, buildings)
    for info_to_retrieve in info_OSM:
        
        # Buildings (all)
        if info_to_retrieve == 'buildings':

            # Retrieve information
            try:
                gdf_buildings = ox.geometries_from_place(place_name, {'building': True})
            
            # No OSM data available
            except:
                print('~ It seems like there is no data available for {} for this island. ~'.format(info_to_retrieve))
                continue

            # Save GeoDataFrame in `island_info`
            island_info['OpenStreetMap']['gdf_buildings'] = gdf_buildings

        # Roads (all)
        elif info_to_retrieve == 'roads':

            # Retrieve information
            try:
                G = ox.graph_from_place(place_name, network_type="all")

            # No OSM data available
            except:
                print('~ It seems like there is no data available for {} for this island. ~'.format(info_to_retrieve))
                continue

            gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

            # Save GeoDataFrame in `island_info`
            island_info['OpenStreetMap']['gdf_roads_nodes'] = gdf_nodes
            island_info['OpenStreetMap']['gdf_roads_edges'] = gdf_edges
    
    return island_info

def exportTIF(island_info, image, path_tif_files=os.path.join(os.getcwd(), 'data', 'tif_files')):

    # Get projection information
    projection = image.select(0).projection().getInfo()
    island_info['spatial_reference']['crs'] = projection['crs']
    island_info['spatial_reference']['crs_transform'] = projection['transform']

    # Export image and save as a .tif file
    geemap.ee_export_image(
        image,
        filename=os.path.join(path_tif_files, 'S2_{}_{}.tif'.format(island_info['general_info']['island'], island_info['general_info']['country'])),
        region=island_info['spatial_reference']['polygon'],
        crs=island_info['spatial_reference']['crs'],
        crs_transform=island_info['spatial_reference']['crs_transform'],
        scale=30)

    # Read that .tif file as a NumPy array
    image_array = tifffile.imread(os.path.join(path_tif_files, 'S2_{}_{}.tif'.format(island_info['general_info']['island'], island_info['general_info']['country'])))

    return island_info, image_array

def plotRGBandOSM(island_info, image_array, plot):

    # Red, green and blue arrays
    red = image_array[:, :, 3]
    green = image_array[:, :, 2]
    blue = image_array[:, :, 1]
    nir = image_array[:, :, 7]

    # RGB & NDVI arrays
    rgb = np.dstack((red, green, blue))
    ndvi = (nir - red) / (nir + red)

    # Get polygon information to build a coordinate extent
    polygon = island_info['spatial_reference']['polygon'].getInfo()
    polygon_array = np.array(polygon['coordinates'][0])

    # Coordinate extent for plotting & consistency
    # TODO: might not be 100% accurate, improve
    longitude_top_left = np.min(polygon_array[:, 0])
    longitude_top_right = np.max(polygon_array[:, 0])
    latitude_bottom_left = np.min(polygon_array[:, 1])
    latitude_top_left = np.max(polygon_array[:, 1])

    # Create the extent
    island_info['spatial_reference']['extent'] = [longitude_top_left, longitude_top_right, latitude_bottom_left, latitude_top_left]

    # Latitude/longitude meshgrid: information for another function
    mg_longitude_image, mg_latitude_image = np.meshgrid(np.linspace(longitude_top_left, longitude_top_right, np.shape(rgb)[0]), \
                                                        np.linspace(latitude_bottom_left, latitude_top_left, np.shape(rgb)[1]))

    island_info['spatial_reference']['meshgrid'] = {'meshgrid_longitude': mg_longitude_image, 'meshgrid_latitude': mg_latitude_image}

    if plot:

        plt.figure()

        # Plot remote sensing image
        # We must divide the array by 255 to obtain the correct range for RGB plotting [0, 255]
        # TODO: image enhancement
        #plt.imshow((rgb/255).astype("uint8"), extent=island_info['spatial_reference']['extent'])#, colormap='gray')
        plt.imshow(ndvi, extent=island_info['spatial_reference']['extent'], cmap='RdYlGn')
        plt.colorbar().set_label('NDVI')

        # Plot roads
        # List of coordinates for roads
        coords_roads = np.array(island_info['OpenStreetMap']['gdf_roads_edges']['geometry'])

        for idx_roads in range(len(coords_roads)):

            if idx_roads == 0:

                plt.plot(*coords_roads[idx_roads].xy, color='white', label='roads')

            else:

                plt.plot(*coords_roads[idx_roads].xy, color='white')

        # Plot buildings
        # List of coordinates for buildings
        coords_buildings = np.array(island_info['OpenStreetMap']['gdf_buildings']['geometry'])

        for idx_buildings in range(len(coords_buildings)):

            if idx_buildings == 0:
                
                if type(coords_buildings[idx_buildings]) == shapely.geometry.point.Point:

                    plt.plot(*coords_buildings[idx_buildings].xy, color='red', label='buildings')
                
                else:
                    
                    plt.plot(*coords_buildings[idx_buildings].exterior.xy, color='red', label='buildings')
            else:
                
                if type(coords_buildings[idx_buildings]) == shapely.geometry.point.Point:

                    plt.plot(*coords_buildings[idx_buildings].xy, color='red')

                else:
                    
                    plt.plot(*coords_buildings[idx_buildings].exterior.xy, color='red')

        #plt.title("NDVI + OpenStreetMap", fontsize=15)
        plt.xlabel("Longitude", fontsize=15)
        plt.ylabel("Latitude", fontsize=15)
        plt.legend(loc=0, fontsize=15)
        plt.savefig('ndvi_osm.png', dpi=300, bbox_inches='tight')
        plt.show()    
    
    return island_info, ndvi

def fillGeometry(island_info, info_OSM, nb_points=30):

    # Loop over OSM information (e.g., roads, buildings)
    for info_to_retrieve in info_OSM:

        # Buildings (all)
        if info_to_retrieve == 'buildings':

            # List of shapely.Polygons for buildings
            coords_buildings = np.array(island_info['OpenStreetMap']['gdf_buildings']['geometry'])
            
            for poly in tqdm(range(len(coords_buildings))):

                # Selecting shapely.Polygon and its coordinates
                polygon = coords_buildings[poly]

                # Building might be a point
                if type(polygon) == shapely.geometry.point.Point:

                    buildings_arr = np.array(polygon.coords)

                    # First iteration, create an array
                    if poly == 0:
                        coords_buildings_filled = buildings_arr
                    
                    # Concanetate new array with previous array
                    else:
                        coords_buildings_filled = np.concatenate((coords_buildings_filled, buildings_arr))
                
                else:

                    buildings_arr = np.array(polygon.exterior.coords)

                    # Generate meshgrid for a square that encompasses the shapely.Polygon
                    mg_lon, mg_lat = np.meshgrid(np.linspace(np.min(buildings_arr[:, 0]), np.max(buildings_arr[:, 0]), nb_points), \
                                                    np.linspace(np.min(buildings_arr[:, 1]), np.max(buildings_arr[:, 1]), nb_points))
                    
                    # Unravel the meshgrid
                    arr_lon = np.ravel(mg_lon)
                    arr_lat = np.ravel(mg_lat)

                    # Create an array of shapely.Point with the arrays of coordinates
                    arr_points = np.array([Point(x, y) for (x, y) in zip(arr_lon, arr_lat)])

                    # Test if the points are in the shapely.Polygon -> return an array of Bool
                    arr_cond = polygon.contains(arr_points)

                    # Combine latitude and longitude that sastisfy the condition in one array
                    arr_points_cond = np.column_stack((arr_lon[np.argwhere(arr_cond)].reshape((-1, )), arr_lat[np.argwhere(arr_cond)].reshape((-1, ))))

                    # If first iteration, create an array
                    if poly == 0:
                        coords_buildings_filled = np.concatenate((buildings_arr, arr_points_cond))
                    
                    # Else, concanetate new array with previous array
                    else:
                        coords_buildings_filled = np.concatenate((coords_buildings_filled, np.concatenate((buildings_arr, arr_points_cond))))
            
            # Save information in dictionary
            island_info['OpenStreetMap']['coords_buildings_filled'] = coords_buildings_filled

        # Roads (all)
        elif info_to_retrieve == 'roads':

            # List of shapely.LineString for roads
            coords_roads = np.array(island_info['OpenStreetMap']['gdf_roads_edges']['geometry'])

            # Loop over all shapely.LineString
            for ls in tqdm(range(len(coords_roads))):

                # Selecting shapely.LineString and its coordinates
                linestring = coords_roads[ls]
                arr_lon = linestring.xy[0]
                arr_lat = linestring.xy[1]

                # Loop over pairs of coordinates within the shapely.LineString
                for pls in range(len(arr_lon)-1):
                    
                    # If first iteration, create an array
                    if ls == 0 and pls == 0:
                        lon_roads_filled = np.linspace(arr_lon[pls], arr_lon[pls+1], nb_points)
                        lat_roads_filled = np.linspace(arr_lat[pls], arr_lat[pls+1], nb_points)

                    # Else, concanetate new array with previous array
                    else:
                        lon_roads_filled = np.concatenate((lon_roads_filled, np.linspace(arr_lon[pls], arr_lon[pls+1], nb_points)))
                        lat_roads_filled = np.concatenate((lat_roads_filled, np.linspace(arr_lat[pls], arr_lat[pls+1], nb_points)))

            # Combine latitude and longitude in one array
            coords_roads_filled = np.column_stack((lon_roads_filled, lat_roads_filled))

            # Save information in dictionary
            island_info['OpenStreetMap']['coords_roads_filled'] = coords_roads_filled
    
    return island_info
            
def generateInfoPreMask(island_info, info_OSM, plot=True):
    
    # Query a recent cloud-free image of the island (Landsat 8)
    image = island_info['image_collection_dict']['S2'].sort("CLOUDY_PIXEL_PERCENTAGE").first()
    island_info, image_array = exportTIF(island_info, image)

    # Plot RGB image with OSM data (optional)
    island_info, ndvi = plotRGBandOSM(island_info, image_array, plot)

    # Fill shapely.LineString and shapely.Polygon (e.g., square to filled square)
    island_info = fillGeometry(island_info, info_OSM)

    return island_info, ndvi

def generateMask(island_info, ndvi, plot=True):

    # Retrieve meshgrids
    mg_longitude_image = island_info['spatial_reference']['meshgrid']['meshgrid_longitude']
    mg_latitude_image = island_info['spatial_reference']['meshgrid']['meshgrid_latitude']

    # Retrieve filled arrays
    coords_roads_filled = island_info['OpenStreetMap']['coords_roads_filled']
    coords_buildings_filled = island_info['OpenStreetMap']['coords_buildings_filled']

    # Create an empty array of the size of the remote sensing image
    #mask = np.zeros((np.shape(mg_longitude_image)[0], np.shape(mg_longitude_image)[1]))
    mask = np.zeros_like(mg_longitude_image)

    # Generate mask for roads
    for coords_road in tqdm(range(np.shape(coords_roads_filled)[0])):

        # Retrieve longitude, latitude
        xx, yy = coords_roads_filled[coords_road, 0], coords_roads_filled[coords_road, 1]

        # Calculate Euclidian distance with the meshgrid
        distance_array = ((xx - mg_longitude_image) ** 2 + (yy - mg_latitude_image) ** 2) ** (1/2)

        # Find the indexes of the closest pixel
        idx = np.unravel_index(np.argmin(distance_array), mg_longitude_image.shape)

        # Assign the closest pixel to 1
        mask[idx[0], idx[1]] = 1

    # Generate mask for buildings
    for coords_building in tqdm(range(np.shape(coords_buildings_filled)[0])):

        # Retrieve longitude, latitude
        xx, yy = coords_buildings_filled[coords_building, 0], coords_buildings_filled[coords_building, 1]

        # Calculate Euclidian distance with the meshgrid
        distance_array = ((xx - mg_longitude_image) ** 2 + (yy - mg_latitude_image) ** 2) ** (1/2)

        # Find the indexes of the closest pixel
        idx = np.unravel_index(np.argmin(distance_array), mg_longitude_image.shape)

        # Assign the closest pixel to 1
        mask[idx[0], idx[1]] = 1

    # Plot result
    if plot:
        plt.figure()
        #plt.imshow(ndvi, extent=island_info['spatial_reference']['extent'], cmap='RdYlGn')
        plt.imshow(np.flip(mask, axis=0), extent=island_info['spatial_reference']['extent'])
        #plt.colorbar()
        plt.xlabel("Longitude", fontsize=15)
        plt.ylabel("Latitude", fontsize=15)
        #plt.title('OSM (roads + buildings) pixelised mask')
        plt.savefig('mask_osm.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Save mask in dictionary
    island_info['OpenStreetMap']['mask_roads_buildings'] = np.flip(mask, axis=0)

    return island_info

def runOSMFunctions(island_info, info_OSM, to_run, island_info_path=os.getcwd()+'\\data\\info_islands'):

    if to_run=='all' or not all(el in to_run for el in ['gdf_buildings', 'gdf_roads_edges', 'gdf_roads_nodes']):
        
        print('~ Querying OpenStreetMap data. ~')

        # Query data from OpenStreetMap
        island_info = queryOpenStreetMap(island_info, info_OSM)

        # Save dictionary
        fw = open(island_info_path + '\\info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country']), 'wb')
        pickle.dump(island_info, fw)
        fw.close()

    if to_run=='all' or not all(el in to_run for el in ['coords_roads_filled', 'coords_buildings_filled']):

        print('~ Filling OpenStreetMap data. ~')

        # Generate roads/buildings information for mask
        island_info, ndvi = generateInfoPreMask(island_info, info_OSM)
        
        # Save dictionary
        fw = open(island_info_path + '\\info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country']), 'wb')
        pickle.dump(island_info, fw)
        fw.close()

    if to_run=='all' or not all(el in to_run for el in ['mask_roads_buildings']):

        print('~ Generating OpenStreetMap mask. ~')

        # Generate roads/buildings mask
        island_info = generateMask(island_info, ndvi)

        # Save dictionary
        fw = open(island_info_path + '\\info_{}_{}.data'.format(island_info['general_info']['island'], island_info['general_info']['country']), 'wb')
        pickle.dump(island_info, fw)
        fw.close()
    
    print('~ All information is available. ~')

    return island_info

def getOpenStreetMap(island, country, info_OSM=['roads', 'buildings'], verbose_init=True, overwrite=True):

    # Retrieve the dictionary with currently available information about the island
    island_info = get_info_islands.retrieve_info_island(island, country, verbose=verbose_init)

    print('\n-------------------------------------------------------------------')
    print('RETRIEVING OpenStreetMap DATA')
    print('Island:', ', '.join([island, country]))
    print('-------------------------------------------------------------------\n')

    # If OpenStreetMap data have NOT already been generated
    if not 'OpenStreetMap' in island_info.keys() or overwrite:

        # Create key/dict for OpenStreetMap data
        island_info['OpenStreetMap'] = {}

        # Run all functions
        island_info = runOSMFunctions(island_info, info_OSM, to_run='all')
    
    # If OpenStreetMap data have already been generated
    else:

        # Make a list of available OSM information
        OSM_info_available = list(island_info['OpenStreetMap'].keys())

        # Run missing functions
        island_info = runOSMFunctions(island_info, info_OSM, to_run=OSM_info_available)

    return island_info