from IslandTime import run_all, retrieve_island_info, save_island_info, update_results_map, update_data_map, PreProcessing, Segmentation, TimeSeriesAnalysis, Metrics
import os
os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '1000'

class Workflow:
    def __init__(self, island, country, run_all=False, overwrite_extract=False, overwrite_preprocess=True, overwrite_analysis=True, execute_segmentation=False, execute_preprocess=False, execute_analysis=False, update_maps=False, small_island=True):
        self.island = island
        self.country = country
        self.run_all = run_all
        self.overwrite_extract = overwrite_extract
        self.overwrite_preprocess = overwrite_preprocess
        self.overwrite_analysis = overwrite_analysis
        self.execute_segmentation = execute_segmentation
        self.execute_preprocess = execute_preprocess
        self.execute_analysis = execute_analysis
        self.update_maps = update_maps
        self.small_island = small_island
        self.path_to_data = os.path.join(os.getcwd(), 'data', 'info_islands')

    def extract_time_series(self, verbose=True):

        # Run all functions
        if self.run_all:
            self.island_info = run_all(self.island, self.country, overwrite=self.overwrite_extract, verbose_init=verbose)

        # Extract island info
        else:
            self.island_info = retrieve_island_info(self.island, self.country, verbose=verbose)

        return self.island_info

    def island_metrics(self, gdf_all_islands=False, overwrite=False):

        # Extract island features
        self.island_info = Metrics(self.island, self.country, gdf_all_islands=gdf_all_islands, overwrite=overwrite).main()

        return self.island_info
    
    def segmentation(self):
            
        # If island info is not available, extract it
        if self.island_info is None:
            self.island_info = self.extract_time_series(verbose=False)

        # Extract coastline time series data using Segmentation
        self.island_info = Segmentation(self.island_info, find_polygons=self.small_island, plot_all=False, time_series_only=True, animation_polygons=False).main()

        # Save island_info
        save_island_info(self.island_info)

        return self.island_info  
     
    def pre_process(self):

        # If island info is not available, extract it
        if self.island_info is None:
            self.island_info = self.extract_time_series(verbose=False)
            
        # Pre-process time series data
        self.island_info = PreProcessing(self.island_info, overwrite=self.overwrite_preprocess).main()

        # Save island_info
        save_island_info(self.island_info)

        return self.island_info

    def analysis(self):
            
        # If island info is not available, extract it
        if self.island_info is None:
            self.island_info = self.extract_time_series(verbose=False)

        # Time series analysis
        self.island_info = TimeSeriesAnalysis(self.island_info, overwrite=self.overwrite_analysis, plot_results_transect=False, plot_only=True, overwrite_transect=False, transect_to_plot=None, overwrite_all=False).main()

        # Save island_info
        save_island_info(self.island_info)

        return self.island_info

    def main(self):
        
        # Extract time series data
        self.extract_time_series()

        # Extract island metrics
        self.island_metrics()

        # Extract coastline time series data using Segmentation
        if self.execute_segmentation:
            self.segmentation()

        # Pre-process time series data
        if self.execute_preprocess:
            self.pre_process()
        
        # Time series analysis
        if self.execute_analysis:
            self.analysis()

        # Update maps
        if self.update_maps:
            # Update result map (all islands)
            update_results_map(self.country, self.path_to_data) 
                    
            # Update availability map (all islands)
            update_data_map(self.path_to_data)
        
        return self.island_info
