# Import modules
from IslandTime import retrieve_island_info, save_island_info
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Polygon, Point
import pickle
import pyproj
import skimage.transform as transform
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '1000'

class Metrics:
    def __init__(self, island=None, country=None, gdf_all_islands=False, overwrite=False, plot_metrics_region=False, region_name=None):

        self.island = island
        self.country = country
        self.gdf_all_islands = gdf_all_islands
        self.overwrite = overwrite
        self.plot_metrics_region = plot_metrics_region
        self.region_name = region_name
        self.gdf_metrics_file = os.path.join(os.getcwd(), 'shp', 'islands_metrics.shp')
        self.path_to_data = os.path.join(os.getcwd(), 'data_example', 'info_islands')

    
    def _calculate_average_curvature(self, polygon: Polygon):
        coords = np.array(polygon.exterior.coords)
        angles = []
        for i in range(len(coords) - 2):
            v1 = coords[i+1] - coords[i]
            v2 = coords[i+2] - coords[i+1]
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            if not np.isnan(angle):
                angles.append(angle)
        
        # Change to degrees
        angles = np.degrees(angles)

        # Calculate mean angle
        mean_angle = np.mean(angles)

        # If mean angle is very small, set it to 0
        return mean_angle if mean_angle >= 1e-6 else 0.
    
    def _calculate_area(self, polygon: Polygon):
        return polygon.area

    def _calculate_elongation(self, polygon: Polygon):
        # Elongation = major axis / minor axis
        # The elongation is a measure of how stretched the shape is.
        # elongation > 1 means the shape is a long and thin shape.
        # elongation < 1 means the shape is a short and fat shape.

        min_rotated_rect = polygon.minimum_rotated_rectangle
        mrr_coords = list(min_rotated_rect.exterior.coords)
        major_axis = max([mrr_coords[i][0] - mrr_coords[i-1][0] for i in range(len(mrr_coords))])
        minor_axis = max([mrr_coords[i][1] - mrr_coords[i-1][1] for i in range(len(mrr_coords))])
        return major_axis / minor_axis

    def _calculate_compactness(self, polygon: Polygon):
        # Compactness = 4 * pi * area / perimeter^2
        # where area is the area of the polygon and perimeter is the length of the polygon
        # The compactness is a measure of how close the shape is to a circle.
        # A compactness of 1 means the shape is a circle, while a compactness of 0 means the shape is a line.
        return (4 * np.pi * polygon.area) / (polygon.length ** 2)

    def _calculate_solidity(self, polygon: Polygon):
        # Solidity = area of polygon / area of convex hull
        # The solidity is a measure of how much of the polygon is filled.
        # A solidity of 1 means the polygon is a solid shape, while a solidity of 0 means the polygon is a line.
        # concave polygons have a solidity < 1, while convex polygons have a solidity = 1.
        # closer to 0, the more concave the polygon is.
        convex_hull = polygon.convex_hull
        return polygon.area / convex_hull.area
    
    def _calculate_sediment_budget(self):

        all_polygons_file = os.path.join(os.getcwd(), 'data_example', 'coastsat_data', '{}_{}'.format(self.island, self.country), 'all_polygons_{}_{}.data'.format(self.island, self.country))
        best_polygons_file = os.path.join(os.getcwd(), 'data_example', 'coastsat_data', '{}_{}'.format(self.island, self.country), 'best_polygons_{}_{}.data'.format(self.island, self.country))

        if os.path.exists(all_polygons_file):
            # Read file
            with open(all_polygons_file, 'rb') as f:
                all_dicts = pickle.load(f)

            # Separate dictionaries
            dict_poly = all_dicts[0]
            dict_rgb = all_dicts[1]
            dict_georef = all_dicts[2]
            dict_image_epsg = all_dicts[3]

        if os.path.exists(best_polygons_file):
            # Read file
            with open(best_polygons_file, 'rb') as f:
                dict_best_polygons = pickle.load(f)

        # Dictionary with corresponding names
        dict_names = {'NIR': 'polygon_NIR_opt',
                        'NDVI': 'polygon_NDVI_opt',
                        'NDWI': 'polygon_NDWI_opt',
                        'NIR Otsu': 'polygon_NIR_otsu',
                        'NDVI Otsu': 'polygon_NDVI_otsu',
                        'NDWI Otsu': 'polygon_NDWI_otsu',
                        'Optimal': 'polygon_opt'}

        area_int_s = []

        for key in dict_best_polygons.keys():
            # Find corresponding polygon
            key_to_name = dict_best_polygons[key]
            polygon = dict_poly[key][dict_names[key_to_name]]

            if polygon is None:
                #print('continue because polygon is None')
                continue

            # Get coordinates of polygon (exterior)
            x_poly, y_poly = polygon.exterior.coords.xy

            # Get image epsg
            image_epsg = dict_image_epsg[key]
            rgb_image = dict_rgb[key]
            georef = dict_georef[key]

            # Define projection
            src_crs = pyproj.CRS('EPSG:3857')
            tgt_crs = pyproj.CRS('EPSG:{}'.format(image_epsg))

            # Create transformer
            transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

            # Transform reference shoreline
            shoreline_x_transformed, shoreline_y_transformed = transformer.transform(self.reference_shoreline[:, 0], self.reference_shoreline[:, 1])
            shoreline_image_crs = np.column_stack((shoreline_x_transformed, shoreline_y_transformed))
            polygon_shoreline_image_crs = Polygon(shoreline_image_crs)
            
            # Make affine transformation matrix
            aff_mat = np.array([[dict_georef[key][1], dict_georef[key][2], dict_georef[key][0]],
                                [dict_georef[key][4], dict_georef[key][5], dict_georef[key][3]],
                                [0, 0, 1]])
            
            # Create affine transformation
            tform = transform.AffineTransform(aff_mat)

            # Transform polygon
            tmp = np.column_stack((x_poly, y_poly))
            points_converted = tform(tmp)
            poly_image_crs = Polygon(points_converted)

            # plt.figure()
            # plt.plot(polygon_shoreline_image_crs.exterior.xy[0], polygon_shoreline_image_crs.exterior.xy[1], color='red', label='Polygon')
            # plt.plot(poly_image_crs.exterior.xy[0], poly_image_crs.exterior.xy[1], color='blue', label='Polygon Image CRS')

            try:
                area_int_s.append((poly_image_crs.area - polygon_shoreline_image_crs.area))
            except Exception as e:
                print(e)
                continue
            
        # remove outliers, NaN, and negative values
        area_int = np.array(area_int_s)
        area_int = area_int[~np.isnan(area_int)]
        area_int = area_int[area_int > 0]

        # remove outliers
        z_scores = np.abs(stats.zscore(area_int))
        area_int = area_int[z_scores < 1.]

        # Calculate mean and std
        mean_area = np.mean(area_int)
        std_area = np.std(area_int)
        median_area = np.median(area_int)
        hist, bin_edges = np.histogram(area_int, bins=20)
        mode_area = np.unique(area_int)[np.argmax(hist)]

        # Calculate 95% confidence interval
        confidence_interval = stats.t.interval(0.95, len(area_int)-1, loc=median_area, scale=std_area/np.sqrt(len(area_int)))

        print('Mean area: {:.2f} m2'.format(mean_area))
        print('Median area: {:.2f} m2'.format(median_area))
        print('Mode area: {:.2f} m2'.format(mode_area))
        print('Standard deviation: {:.2f} m2'.format(std_area))

        # plot results using seaborn
        # sns.set(style="whitegrid")
        # plt.figure(figsize=(10, 6))
        # plt.hist(area_int, bins=30, color='blue', alpha=0.7, edgecolor='black')
        # plt.axvline(median_area, color='red', linestyle='dashed', linewidth=1, label='Mean: {:.2f}'.format(median_area))
        # plt.axvline(confidence_interval[0], color='green', linestyle='dashed', linewidth=1, label='95% CI: {:.2f}'.format(confidence_interval[0]))
        # plt.axvline(confidence_interval[1], color='green', linestyle='dashed', linewidth=1, label='95% CI: {:.2f}'.format(confidence_interval[1]))
        # plt.title('Histogram of Area of Intersection between Polygon and Reference Shoreline')
        # plt.xlabel('Area (m²)')
        # plt.ylabel('Frequency')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # # plot results
        # plt.figure(figsize=(10, 6))
        # plt.plot(area_int, color='blue', alpha=0.5, label='Area of Intersection', marker='o', markersize=3, linestyle=' ')
        # plt.axhline(median_area, color='red', linestyle='dashed', linewidth=1, label='Mean: {:.2f}'.format(median_area))
        # plt.axhline(confidence_interval[0], color='green', linestyle='dashed', linewidth=1, label='95% CI: {:.2f}'.format(confidence_interval[0]))
        # plt.axhline(confidence_interval[1], color='green', linestyle='dashed', linewidth=1, label='95% CI: {:.2f}'.format(confidence_interval[1]))
        # plt.title('Area of Intersection between Polygon and Reference Shoreline')
        plt.xlabel('Area (m²)')

        return mean_area, std_area, median_area, mode_area, confidence_interval

    def calculate_all_metrics(self, island, country, all_islands: bool, point: Point, polygon: Polygon):
        # Calculate metrics
        area = self._calculate_area(polygon)
        compactness = self._calculate_compactness(polygon)
        solidity = self._calculate_solidity(polygon)
        elongation = self._calculate_elongation(polygon)
        avg_curvature = self._calculate_average_curvature(polygon)
        sb_mean_area, sb_std_area, sb_median_area, sb_mode_area, sb_ci = self._calculate_sediment_budget()

        # Return dictionary with metrics
        if all_islands:
            return {
                'island': island,
                'country': country,
                'geometry': point,
                'area': area,
                'compact': compactness,
                'solidity': solidity,
                'elongation': elongation,
                'avg_curv': avg_curvature,
                'sediment_budget_mean_area': sb_mean_area,
                'sediment_budget_std_area': sb_std_area,
                'sediment_budget_median_area': sb_median_area,
                'sediment_budget_mode_area': sb_mode_area,
                'sediment_budget_ci': sb_ci
            }, {
                'area': area,
                'compactness': compactness,
                'solidity': solidity,
                'elongation': elongation,
                'avg_curvature': avg_curvature,
                'sediment_budget_mean_area': sb_mean_area,
                'sediment_budget_std_area': sb_std_area,
                'sediment_budget_median_area': sb_median_area,
                'sediment_budget_mode_area': sb_mode_area,
                'sediment_budget_ci': sb_ci
            }
        
        else:
            return {
                'area': area,
                'compactness': compactness,
                'solidity': solidity,
                'elongation': elongation,
                'avg_curvature': avg_curvature,
                'sediment_budget_mean_area': sb_mean_area,
                'sediment_budget_std_area': sb_std_area,
                'sediment_budget_median_area': sb_median_area,
                'sediment_budget_mode_area': sb_mode_area,
                'sediment_budget_ci': sb_ci
            }


    def f_plot_metrics_region(self, region_name):

        print('Getting island metrics inside {}'.format(region_name))

        # get the list of islands inside the region
        islands_shp = 'excel//{}_islands.xlsx'.format(region_name.replace(' ', '_'))
        if os.path.exists(islands_shp):
            islands_region = pd.read_excel(islands_shp)
        
        else:
            raise ValueError('Please run the list_islands_inside_region function first (IslandTimePlotting)')

        # create empty list to store data
        data_list = []
        island_list = []

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
            
            try:
                island_metrics = island_info['island_metrics']
            
            except Exception as e:
                print(f'Error with island: {island} - {str(e)}')
                continue

            # the output is a dictionary with the metrics, transform it to a dataframe
            data_list.append(island_metrics) 
            island_list.append(island)

        return pd.DataFrame(data_list, index=island_list)

    def main(self):
        
        print('\n-------------------------------------------------------------------')
        print('Calculating island metrics (area, compactness, solidity, elongation, average curvature)')
        print('Island: {}'.format(self.island))
        print('-------------------------------------------------------------------\n')

        if self.plot_metrics_region:
            df_island_metrics = self.f_plot_metrics_region(self.region_name)
            return df_island_metrics

        else:
            # Only calculate metrics for the specified island
            if not self.gdf_all_islands:

                try:
                    island_info = retrieve_island_info(self.island, self.country, verbose=False)
                
                except Exception as e:
                    print(f'Error with island: {self.island} - {str(e)}')
                    return None
                
                # Check if metrics already exist
                if 'island_metrics' in island_info and not self.overwrite:
                    print(f'Metrics already exist for island: {self.island}')
                    return island_info

                # Extract latitude, longitude and reference shoreline
                latitude, longitude = island_info['spatial_reference']['latitude'], island_info['spatial_reference']['longitude']
                reference_shoreline = island_info['spatial_reference']['reference_shoreline']
                self.reference_shoreline = reference_shoreline

                # Create point and polygon
                point = Point(longitude, latitude)
                polygon = Polygon(reference_shoreline)

                # Calculate metrics
                dict_metrics = self.calculate_all_metrics(self.island, self.country, all_islands=False, point=point, polygon=polygon)

                # Save to island_info
                island_info['island_metrics'] = dict_metrics

                # Save new island_info
                save_island_info(island_info)

                return island_info
            
            # Calculate metrics for all islands
            else:
                # Open existing GeoDataFrame if it exists
                if os.path.exists(self.gdf_metrics_file) and not self.overwrite:
                    gdf_metrics_existing = gpd.read_file(self.gdf_metrics_file)
                    list_islands = gdf_metrics_existing['island'].values
                
                else:
                    gdf_metrics_existing = None
                    list_islands = []

                # get the list of islands inside the region
                islands_shp = 'excel//{}_islands.xlsx'.format(self.region_name.replace(' ', '_'))
                if os.path.exists(islands_shp):
                    islands_region = pd.read_excel(islands_shp)
                
                else:
                    raise ValueError('Please run the list_islands_inside_region function first (IslandTimePlotting)')
                
                # Create empty list to store data
                data_list = []

                # Loop through all islands
                for file in tqdm(os.listdir(self.path_to_data)):
                    if file == 'archives':
                        continue
                    island = file.split('_')[1] 
                    country = file.split('_')[2].split('.')[0]


                    if island in list_islands:
                        continue

                    if not island in islands_region['island'].values:
                        continue


                    try:
                        island_info = retrieve_island_info(island, country, verbose=False)
                    
                    except Exception as e:
                        print(f'Error with island: {island} - {str(e)}')
                        continue

                    if 'reference_shoreline' not in island_info['spatial_reference'].keys():
                        continue

                    if 'timeseries_analysis' not in island_info.keys():
                        continue
                    
                    # Extract latitude, longitude and reference shoreline
                    latitude, longitude = island_info['spatial_reference']['latitude'], island_info['spatial_reference']['longitude']
                    reference_shoreline = island_info['spatial_reference']['reference_shoreline']

                    # Create point and polygon
                    point = Point(longitude, latitude)
                    polygon = Polygon(reference_shoreline)

                    # Calculate metrics
                    try:
                        dict_metrics, dict_metrics_island = self.calculate_all_metrics(island=island, country=country, all_islands=True, point=point, polygon=polygon)
                    
                    except Exception as e:
                        print(f'Error with island: {island} - {str(e)}')
                        continue

                    # Save to island_info
                    island_info['island_metrics'] = dict_metrics_island

                    # Save new island_info
                    save_island_info(island_info)

                    # Add data to list
                    data_list.append(dict_metrics)

                # Create GeoDataFrame
                if len(data_list) > 0:
                    gdf_metrics = gpd.GeoDataFrame(data_list, crs='EPSG:4326')
                
                else:
                    gdf_metrics = None

                # Combine with existing GeoDataFrame
                if gdf_metrics_existing is not None:
                    gdf_metrics_final = pd.concat([gdf_metrics_existing, gdf_metrics], ignore_index=True)
                
                elif gdf_metrics is None:
                    gdf_metrics_final = gdf_metrics_existing
                
                else:
                    gdf_metrics_final = gdf_metrics

                # Save to shapefile
                gdf_metrics_final.to_file(self.gdf_metrics_file)

                return gdf_metrics_final