from IslandTime import run_all, retrieve_island_info, save_island_info, PreProcessing, Segmentation, TimeSeriesAnalysis

class Workflow:
    def __init__(self, island, country, run_all=False, overwrite_extract=False, overwrite_preprocess=True, overwrite_analysis=True, execute_segmentation=False, execute_preprocess=False, execute_analysis=True):
        self.island = island
        self.country = country
        self.run_all = run_all
        self.overwrite_extract = overwrite_extract
        self.overwrite_preprocess = overwrite_preprocess
        self.overwrite_analysis = overwrite_analysis
        self.execute_segmentation = execute_segmentation
        self.execute_preprocess = execute_preprocess
        self.execute_analysis = execute_analysis

    def extract_time_series(self, verbose=True):

        # Run all functions
        if self.run_all:
            self.island_info = run_all(self.island, self.country, overwrite=self.overwrite_extract, verbose_init=verbose)

        # Extract island info
        else:
            self.island_info = retrieve_island_info(self.island, self.country, verbose=verbose)

        return self.island_info
    
    def segmentation(self):
            
        # If island info is not available, extract it
        if self.island_info is None:
            self.island_info = self.extract_time_series(verbose=False)

        # Extract coastline time series data using Segmentation
        self.island_info = Segmentation(self.island_info).main()

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
        self.island_info = TimeSeriesAnalysis(self.island_info, overwrite=self.overwrite_analysis).main()

        # Save island_info
        save_island_info(self.island_info)

        return self.island_info

    def main(self):
        
        # Extract time series data
        self.extract_time_series()

        # Extract coastline time series data using Segmentation
        if self.execute_segmentation:
            self.segmentation()

        # Pre-process time series data
        if self.execute_preprocess:
            self.pre_process()
        
        # Time series analysis
        if self.execute_analysis:
            self.analysis()
        
        # Update availability map (all islands)
        #self.update_availability_map()
        
        # Update result map (all islands)
        #self.update_result_map()
        
        return self.island_info
