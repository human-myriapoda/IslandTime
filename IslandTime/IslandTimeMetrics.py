# Import modules
from IslandTime import retrieve_island_info, save_island_info
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Polygon, Point
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '1000'

class Metrics:
    def __init__(self, island=None, country=None, gdf_all_islands=False, overwrite=False):
        self.island = island
        self.country = country
        self.gdf_all_islands = gdf_all_islands
        self.overwrite = overwrite
        self.gdf_metrics_file = os.path.join(os.getcwd(), 'shp', 'islands_metrics.shp')
        self.path_to_data = os.path.join(os.getcwd(), 'data', 'info_islands')
    
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
        min_rotated_rect = polygon.minimum_rotated_rectangle
        mrr_coords = list(min_rotated_rect.exterior.coords)
        major_axis = max([mrr_coords[i][0] - mrr_coords[i-1][0] for i in range(len(mrr_coords))])
        minor_axis = max([mrr_coords[i][1] - mrr_coords[i-1][1] for i in range(len(mrr_coords))])
        return major_axis / minor_axis

    def _calculate_compactness(self, polygon: Polygon):
        return (4 * np.pi * polygon.area) / (polygon.length ** 2)

    def _calculate_solidity(self, polygon: Polygon):
        convex_hull = polygon.convex_hull
        return polygon.area / convex_hull.area

    def calculate_all_metrics(self, island, country, all_islands: bool, point: Point, polygon: Polygon):
        # Calculate metrics
        area = self._calculate_area(polygon)
        compactness = self._calculate_compactness(polygon)
        solidity = self._calculate_solidity(polygon)
        elongation = self._calculate_elongation(polygon)
        avg_curvature = self._calculate_average_curvature(polygon)

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
                'avg_curv': avg_curvature
            }, {
                'area': area,
                'compactness': compactness,
                'solidity': solidity,
                'elongation': elongation,
                'avg_curvature': avg_curvature
            }
        
        else:
            return {
                'area': area,
                'compactness': compactness,
                'solidity': solidity,
                'elongation': elongation,
                'avg_curvature': avg_curvature
            }
        
    def main(self):
        
        print('\n-------------------------------------------------------------------')
        print('Calculating island metrics (area, compactness, solidity, elongation, average curvature)')
        print('-------------------------------------------------------------------\n')

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
            
            # Create empty list to store data
            data_list = []

            # Loop through all islands
            for file in tqdm(os.listdir(self.path_to_data)):
                island = file.split('_')[1] 
                country = file.split('_')[2].split('.')[0]

                if island in list_islands:
                    continue

                try:
                    island_info = retrieve_island_info(island, country, verbose=False)
                
                except Exception as e:
                    print(f'Error with island: {island} - {str(e)}')
                    continue

                if 'reference_shoreline' not in island_info['spatial_reference'].keys():
                    continue
                
                # Extract latitude, longitude and reference shoreline
                latitude, longitude = island_info['spatial_reference']['latitude'], island_info['spatial_reference']['longitude']
                reference_shoreline = island_info['spatial_reference']['reference_shoreline']

                # Create point and polygon
                point = Point(longitude, latitude)
                polygon = Polygon(reference_shoreline)

                # Calculate metrics
                dict_metrics, dict_metrics_island = self.calculate_all_metrics(island=island, country=country, all_islands=True, point=point, polygon=polygon)

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