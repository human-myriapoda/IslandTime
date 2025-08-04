import pyproj
import pandas as pd
import osmnx as ox
import shapely
import os
from tqdm import tqdm
from IslandTime import retrieve_island_info
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# TODO: optimise

class RegionPlotting:
    def __init__(self, region_name, path_to_data=os.path.join(os.getcwd(), 'data', 'info_islands')):
        self.region_name = region_name
        self.path_to_data = path_to_data
    
    def _get_polygon_region_osmnx(self, region_name:str):
        # query OSMnx for the region
        region = ox.geocode_to_gdf('{}, Maldives'.format(region_name))

        # get the bounding box of the region
        region_box = [region.bbox_north.values[0], \
                        region.bbox_west.values[0], \
                        region.bbox_south.values[0], \
                        region.bbox_east.values[0]]

        # create polygon with the bounding box
        region_polygon = shapely.geometry.Polygon([(region_box[1], region_box[0]), \
                                                        (region_box[1], region_box[2]), \
                                                        (region_box[3], region_box[2]), \
                                                        (region_box[3], region_box[0])])
        
        return region_polygon

    def list_islands_inside_region(self, region_name:str, region_shp=None, use_osmnx=False):

        print('Getting the islands inside {}'.format(region_name))

        # get the region polygon
        if use_osmnx and region_shp is None:
            region_polygon = self._get_polygon_region_osmnx(region_name)
        
        elif region_shp is None:
            raise ValueError('Please provide a shapefile for the region')
        
        else:
            region_shp_file = gpd.read_file(region_shp)
            region_polygon = region_shp_file.geometry[0]

        # create empty list to store data
        data_list = []

        # iterate over the files in the folder
        for file_isl in tqdm(os.listdir(self.path_to_data)):

            if file_isl == 'archives':
                continue

            # get the island name and country
            island = file_isl.split('_')[1] 
            country = file_isl.split('_')[2].split('.')[0]

            try:
                island_info = retrieve_island_info(island, country, verbose=False)
            
            except Exception as e:
                print(f'Error with island: {island} - {str(e)}')
                continue
            
            # get the coordinates of the island and create a point
            lat, lon = island_info['spatial_reference']['latitude'], island_info['spatial_reference']['longitude']
            point = shapely.geometry.Point(lon, lat)

            # check if the point is inside the region
            if region_polygon.contains(point):
                data_list.append(
                    {
                        'island': island,
                        'country': country,
                        'geometry': point
                    }
                )

        # create a geodataframe
        gdf_islands = gpd.GeoDataFrame(data_list, crs='EPSG:4326')

        # save the geodataframe
        gdf_islands.to_file('shp//{}_islands.shp'.format(region_name.replace(' ', '_')))

        # create a dataframe
        df_islands = pd.DataFrame(data_list)

        # save the dataframe
        df_islands.to_excel('excel//{}_islands.xlsx'.format(region_name.replace(' ', '_')))
    
    def seasonal_results_inside_region(self, region_name, ncond_seasonality):

        print('Getting the seasonal results inside {}'.format(region_name))

        # get the list of islands inside the region
        islands_shp = 'excel//{}_islands.xlsx'.format(region_name.replace(' ', '_'))
        if os.path.exists(islands_shp):
            islands_region = pd.read_excel(islands_shp)
        
        else:
            raise ValueError('Please run the list_islands_inside_region function first')

        # create empty list to store data
        data_list = []

        # iterate over the files in the folder
        for file_isl in tqdm(os.listdir(self.path_to_data)):

            if file_isl == 'archives':
                continue

            # get the island name and country
            island = file_isl.split('_')[1]
            country = file_isl.split('_')[2].split('.')[0]

            if not island in islands_region['island'].values:
                continue

            try:
                island_info = retrieve_island_info(island, country, verbose=False)

            except Exception as e:
                print(f'Error with island: {island} - {str(e)}')
                continue
            
            if 'timeseries_analysis' in island_info.keys():
                if island_info['timeseries_analysis'] != {}:
                
                    # get the coordinates of the island
                    lon, lat = island_info['spatial_reference']['longitude'], island_info['spatial_reference']['latitude']
                    
                    # Extract reference shoreline and transects
                    reference_shoreline = island_info['spatial_reference']['reference_shoreline']
                    transects = island_info['spatial_reference']['transects']

                    # Create a polygon from the reference shoreline
                    polygon_reference_shoreline = shapely.geometry.Polygon(reference_shoreline)

                    # Short cut for the time series analysis results
                    ts_analysis_results = island_info['timeseries_analysis']

                    # Extract the key of the transects
                    key_transects = [int((key).split('_')[3]) for key in ts_analysis_results.keys()]

                    # Extract the intersection between the reference shoreline and the transects
                    x_intersections, y_intersections = [], []
                    try:
                        intersections = [polygon_reference_shoreline.exterior.intersection(shapely.geometry.LineString(transects[key_transect])) for key_transect in key_transects]
                    except:
                        continue

                    # extract the coordinates of the intersections
                    for intersection in intersections:
                        if type(intersection) == shapely.geometry.MultiPoint:
                            # Take the first point of the MultiPoint
                            x_intersections.append(intersection.geoms[0].x)
                            y_intersections.append(intersection.geoms[0].y)

                        elif type(intersection) == shapely.geometry.LineString:
                            x_intersections.append(0.)
                            y_intersections.append(0.)

                        elif type(intersection) == shapely.geometry.collection.GeometryCollection:
                            x_intersections.append(0.)
                            y_intersections.append(0.)

                        else:
                            x_intersections.append(intersection.x)
                            y_intersections.append(intersection.y)

                    # Reproject the coordinates to the Web Mercator projection
                    # Source and target coordinate reference systems
                    tgt_crs = pyproj.CRS('EPSG:4326')
                    src_crs = pyproj.CRS('EPSG:3857')

                    # Define transformer
                    transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

                    # # Transform latitude and longitude to x and y
                    x_intersections, y_intersections = transformer.transform(x_intersections, y_intersections)

                    # Transform reference shoreline to x and y
                    x_reference_shoreline, y_reference_shoreline = transformer.transform(polygon_reference_shoreline.exterior.coords.xy[0], polygon_reference_shoreline.exterior.coords.xy[1])

                    # Trend splot and results (colorbar = trend slope and symbol = trend result)
                    c_trend = [ts_analysis_results[val]['trend']['trend_slope'] for val in ts_analysis_results.keys()]
                    symbols_trend = [ts_analysis_results[val]['trend']['trend_result'] for val in ts_analysis_results.keys()]

                    # Seasonality amplitude results (colorbar = seasonality amplitude)
                    if 'amplitude_seasonal_BEAST_absrange' in ts_analysis_results[list(ts_analysis_results.keys())[0]]['seasonality'].keys():
                        c_seasonality_amplitude = [ts_analysis_results[val]['seasonality']['amplitude_seasonal_STL_absrange'] for val in ts_analysis_results.keys()]
                        c_seasonality_amplitude_BEAST = [ts_analysis_results[val]['seasonality']['amplitude_seasonal_BEAST_absrange'] for val in ts_analysis_results.keys()]

                    else:
                        c_seasonality_amplitude = [0. for val in ts_analysis_results.keys()]

                    # Seasonality peaks and minima results
                    if 'conditions_seasonality' in ts_analysis_results[list(ts_analysis_results.keys())[0]]['seasonality'].keys():
                        c_seasonality_peaks = [ts_analysis_results[val]['seasonality']['seasonality_indian_monsoon_peaks'] if sum(ts_analysis_results[val]['seasonality']['conditions_seasonality'].values()) >= ncond_seasonality else 'undetermined' for val in ts_analysis_results.keys()]
                        c_seasonality_minima = [ts_analysis_results[val]['seasonality']['seasonality_indian_monsoon_minima'] if sum(ts_analysis_results[val]['seasonality']['conditions_seasonality'].values()) >= ncond_seasonality else 'undetermined' for val in ts_analysis_results.keys()]
                        c_seasonality_peaks_BEAST = [ts_analysis_results[val]['seasonality']['seasonality_indian_monsoon_peaks_BEAST'] if sum(ts_analysis_results[val]['seasonality']['conditions_seasonality'].values()) >= ncond_seasonality else 'undetermined' for val in ts_analysis_results.keys()]
                        c_seasonality_minima_BEAST = [ts_analysis_results[val]['seasonality']['seasonality_indian_monsoon_minima_BEAST'] if sum(ts_analysis_results[val]['seasonality']['conditions_seasonality'].values()) >= ncond_seasonality else 'undetermined' for val in ts_analysis_results.keys()]
                    
                    else:
                        continue
                        
                    # Add data to list
                    for x in range(len(x_intersections)):
                        data_list.append({
                            'island': island,
                            'country': country,
                            'geometry': shapely.geometry.Point(x_intersections[x], y_intersections[x]),
                            'aSTL': c_seasonality_amplitude[x],
                            'aBEAST': c_seasonality_amplitude_BEAST[x],
                            'pSTL': c_seasonality_peaks[x],
                            'mSTL': c_seasonality_minima[x],
                            'pBEAST': c_seasonality_peaks_BEAST[x],
                            'mBEAST': c_seasonality_minima_BEAST[x]
                        })

        # Create a geodataframe
        gdf = gpd.GeoDataFrame(data_list, crs='EPSG:4326')

        # Save the geodataframe
        gdf.to_file('shp//{}_seasonality_cond{}_results.shp'.format(region_name.replace(' ', '_'), ncond_seasonality))

    # calculate orientation of each transect
    def _calculate_orientation(self, transect):
        x1, y1 = transect.coords[0]
        x2, y2 = transect.coords[-1]
        angle = np.arctan2(y2 - y1, x2 - x1)

        # in positive degree, 0 to 360
        angle_deg = np.degrees(angle) % 360

        # nautical orientation
        angle_deg = (90 - angle_deg) % 360

        return angle_deg

    def rose_diagram_min_peak(self, region_name, ncond_seasonality, feature_res='mBEAST'):

        print(f'Plotting the rose diagram for {region_name} for {feature_res} for {ncond_seasonality} conditions met')

        # Query geodataframe of results and list of islands
        gdf_results = gpd.read_file(f'shp//{region_name.replace(" ", "_")}_seasonality_cond{ncond_seasonality}_results.shp')
        list_islands = pd.read_excel(f'excel//{region_name.replace(" ", "_")}_islands.xlsx')['island'].values

        # Query reference shorelines
        reference_shorelines = gpd.read_file('shp//islands_reference_shoreline.shp')

        # Select island columns in gdf
        gdf_isl = gdf_results[gdf_results['island'].isin(list_islands)].copy()  # Explicitly create a copy

        # Select reference shorelines
        rs_isl = reference_shorelines[reference_shorelines['island'].isin(list_islands)].copy()  # Explicitly create a copy

        # Define the projection and transformer
        src_crs = pyproj.CRS('EPSG:3857')
        tgt_crs = pyproj.CRS('EPSG:4326')
        transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

        # Initialize an empty DataFrame for merged results
        gdf_isl_merged = pd.DataFrame()

        # Get transects
        for isl in tqdm(list_islands):
            # Retrieve island information
            island_info = retrieve_island_info(isl, 'Maldives', verbose=False)
            if 'transects' not in island_info['spatial_reference'].keys():
                print(f'Transects not found for {isl}')
                continue
            transects = island_info['spatial_reference']['transects']

            # Subset gdf and explicitly create a copy
            gdf_isl_subset = gdf_isl[gdf_isl['island'] == isl].copy()
            # Transform transects
            transects_image_crs = {}
            for key_transect, transect in transects.items():
                transect_x_transformed, transect_y_transformed = transformer.transform(transect[:, 0], transect[:, 1])
                transects_image_crs[key_transect] = np.column_stack((transect_x_transformed, transect_y_transformed))

            # Create dictionary of LineString
            transects_linestring = {}
            for key_transect, transect in transects_image_crs.items():
                transects_linestring[key_transect] = shapely.geometry.LineString(transect)

            # Calculate orientation of each transect
            transects_orientation = {}
            for key_transect, transect in transects_linestring.items():
                transects_orientation[key_transect] = self._calculate_orientation(transect)

            # Find closest transect to points
            gdf_isl_subset['closest_transect'] = gdf_isl_subset['geometry'].apply(lambda x: min(transects_linestring, key=lambda y: transects_linestring[y].distance(x)))

            # Calculate orientation of each point
            gdf_isl_subset['orientation'] = gdf_isl_subset['closest_transect'].map(transects_orientation)

            # Define orientation bins
            num_bins = 36
            bin_edges = np.linspace(0, 360, num_bins + 1)

            # Bin the orientations
            gdf_isl_subset['orientation_bin'] = pd.cut(gdf_isl_subset['orientation'], bins=bin_edges, include_lowest=True, labels=False)

            # Merge gdf with the next iteration
            if isl == list_islands[0]:
                gdf_isl_merged = gdf_isl_subset.copy()  # Ensure gdf_isl_merged is a copy
            else:
                gdf_isl_merged = pd.concat([gdf_isl_merged, gdf_isl_subset])

        # Group by orientation bin and category
        grouped = gdf_isl_merged.groupby(['orientation_bin', feature_res]).size().unstack(fill_value=0)

        # Convert bin numbers to radians
        bin_centers = np.deg2rad(bin_edges[:-1] + np.diff(bin_edges) / 2)

        # Create custom colormap
        dict_colours_labels = {'sw_monsoon': '#58508d',
                                'ne_monsoon': '#bc5090', 
                                'transition_sw_ne': '#ffa600',
                                'transition_ne_sw': '#ff6361',
                                'undetermined': '#003f5c'}

        dict_labels = {'sw_monsoon': 'SW Monsoon',
                        'ne_monsoon': 'NE Monsoon',
                        'transition_sw_ne': 'SW-NE Transition',
                        'transition_ne_sw': 'NE-SW Transition',
                        'undetermined': 'Undetermined'}

        colors = [dict_colours_labels[col] for col in grouped.columns]

        fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True), figsize=(10, 6))

        # Plot each category as a stacked bar
        bottom = np.zeros(num_bins)

        for i, category in enumerate(grouped.columns):
            values = grouped[category].reindex(range(num_bins), fill_value=0).values
            bars = ax.bar(bin_centers, values, width=np.deg2rad(np.diff(bin_edges)), bottom=bottom, color=colors[i], edgecolor='black', label=dict_labels[category], zorder=3)
            bottom += values
        ax.grid(zorder=0)

        # Increase tick size & bold
        ax.tick_params(axis='both', which='major', labelsize=15, colors='white')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
    
        fig.savefig(f'figures//atoll//{region_name}//rose_diagram_{region_name.replace(" ", "_")}_cond{ncond_seasonality}_{feature_res}.png', dpi=300, bbox_inches='tight', transparent=True)

    def rose_diagram_amplitude(self, region_name, ncond_seasonality, feature_res='aBEAST'):
    
        print(f'Plotting the rose diagram for {region_name} for {feature_res} for {ncond_seasonality} conditions met')

        # Query geodataframe of results and list of islands
        gdf_results = gpd.read_file(f'shp//{region_name.replace(" ", "_")}_seasonality_cond{ncond_seasonality}_results.shp')
        list_islands = pd.read_excel(f'excel//{region_name.replace(" ", "_")}_islands.xlsx')['island'].values

        # Query reference shorelines
        reference_shorelines = gpd.read_file('shp//islands_reference_shoreline.shp')

        # Select island columns in gdf
        gdf_isl = gdf_results[gdf_results['island'].isin(list_islands)].copy()  # Explicitly create a copy

        # Select reference shorelines
        rs_isl = reference_shorelines[reference_shorelines['island'].isin(list_islands)].copy()  # Explicitly create a copy

        # Define the projection and transformer
        src_crs = pyproj.CRS('EPSG:3857')
        tgt_crs = pyproj.CRS('EPSG:4326')
        transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

        # Define amplitude categories
        amplitude_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, np.inf]
        amplitude_labels = ['<5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '>40']

        # Initialize merged DataFrame
        gdf_isl_merged = pd.DataFrame()

        # Loop through each island
        for isl in tqdm(list_islands):
            # Retrieve island information
            island_info = retrieve_island_info(isl, 'Maldives', verbose=False)
            if 'transects' not in island_info['spatial_reference'].keys():
                print(f'Transects not found for {isl}')
                continue
            transects = island_info['spatial_reference']['transects']

            # Filter DataFrame for the current island
            gdf_isl_subset = gdf_isl[gdf_isl['island'] == isl].copy()

            # Transform transects
            transects_image_crs = {}
            for key_transect, transect in transects.items():
                transect_x_transformed, transect_y_transformed = transformer.transform(
                    transect[:, 0], transect[:, 1]
                )
                transects_image_crs[key_transect] = np.column_stack(
                    (transect_x_transformed, transect_y_transformed)
                )

            # Create LineString objects
            transects_linestring = {
                key: shapely.geometry.LineString(transect)
                for key, transect in transects_image_crs.items()
            }

            # Calculate orientations
            transects_orientation = {
                key: self._calculate_orientation(transect)
                for key, transect in transects_linestring.items()
            }

            # Find closest transect to each point
            gdf_isl_subset.loc[:, 'closest_transect'] = gdf_isl_subset['geometry'].apply(
                lambda x: min(transects_linestring, key=lambda y: transects_linestring[y].distance(x))
            )

            # Assign orientations based on closest transects
            gdf_isl_subset.loc[:, 'orientation'] = gdf_isl_subset['closest_transect'].map(
                transects_orientation
            )

            # Classify amplitudes into bins
            gdf_isl_subset.loc[:, 'amplitude_category'] = pd.cut(
                gdf_isl_subset[feature_res], bins=amplitude_bins, labels=amplitude_labels
            )

            # Append to the merged DataFrame
            gdf_isl_merged = pd.concat([gdf_isl_merged, gdf_isl_subset], ignore_index=True)

        # Define orientation bins
        num_bins = 36
        bin_edges = np.linspace(0, 360, num_bins + 1)

        # Bin orientations
        gdf_isl_merged.loc[:, 'orientation_bin'] = pd.cut(
            gdf_isl_merged['orientation'], bins=bin_edges, include_lowest=True, labels=False
        )

        # Group by orientation bin and amplitude category
        grouped = gdf_isl_merged.groupby(['orientation_bin', 'amplitude_category']).size().unstack(
            fill_value=0
        )

        # Convert bin numbers to radians
        bin_centers = np.deg2rad(bin_edges[:-1] + np.diff(bin_edges) / 2)

        # Define colors for each category
        colors = plt.cm.plasma(np.linspace(0, 1, len(grouped.columns)))

        # Plot the rose diagram
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True), figsize=(10, 6))

        # Plot each amplitude category as a stacked bar
        bottom = np.zeros(num_bins)
        for i, category in enumerate(grouped.columns):
            values = grouped[category].reindex(range(num_bins), fill_value=0).values
            ax.bar(
                bin_centers,
                values,
                width=np.deg2rad(np.diff(bin_edges)),
                bottom=bottom,
                color=colors[i],
                edgecolor='black',
                label=category,
                zorder=3,
            )
            bottom += values

        ax.grid(zorder=0)
        ax.tick_params(axis='both', which='major', labelsize=15, colors='white')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        fig.savefig(f'figures//atoll//{region_name}//rose_diagram_{region_name.replace(" ", "_")}_ncond{ncond_seasonality}_{feature_res}.png',
                    dpi=300, 
                    bbox_inches='tight', 
                    transparent=True)

    def plotting_workflow_region(self):

        if not os.path.exists(f'figures//atoll//{self.region_name}'):
            os.makedirs(f'figures//atoll//{self.region_name}')

        list_conditions = [2, 3]

        for ncond in list_conditions:
            # check if the list of islands inside the region exists
            if not os.path.exists(f'excel//{self.region_name.replace(" ", "_")}_islands.xlsx'):
                self.list_islands_inside_region(self.region_name, region_shp='shp_atolls//{}.shp'.format(self.region_name.replace(" ", "_")), use_osmnx=False)

            # check if the seasonal results inside the region exists
            if not os.path.exists(f'shp//{self.region_name.replace(" ", "_")}_seasonality_cond{ncond}_results.shp'):
                self.seasonal_results_inside_region(self.region_name, ncond)
            
            list_features = ['mBEAST', 'pBEAST']
            # plot the rose diagram
            for feature in list_features:
                self.rose_diagram_min_peak(self.region_name, ncond, feature_res=feature)
        
            # plot the classified amplitude rose diagram
            self.rose_diagram_amplitude(self.region_name, ncond)
