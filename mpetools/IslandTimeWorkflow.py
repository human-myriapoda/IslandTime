
from mpetools import run_all, retrieve_island_info, PreProcess

class Workflow:
    def __init__(self, island, country, run_all=False, overwrite=False):
        self.island = island
        self.country = country
        self.run_all = run_all
        self.overwrite = overwrite

    def extract_time_series(self):

        # Run all functions
        if self.run_all:
            island_info = run_all(self.island, self.country, overwrite=self.overwrite)

        # Extract island info
        else:
            island_info = retrieve_island_info(self.island, self.country)

        return island_info
    
    def pre_process(self):

        # If island info is not available, extract it
        if self.island_info is None:
            self.island_info = self.extract_time_series()
            
        # Pre-process time series data
        island_info = PreProcess(self.island_info, overwrite=self.overwrite).main()

        return island_info
    
    def main(self):
        
        # Extract time series data
        self.island_info = self.extract_time_series()

        # Extract coastline time series data using Segmentation

        # Pre-process time series data
        self.island_info = self.pre_process()

        return self.island_info
