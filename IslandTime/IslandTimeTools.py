"""
List of tools for the IslandTime workflow.

Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
from IslandTime import PreTimeSeries, IslandTimeBase
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt

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
        print('Retrieving all information available for the island')
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

def save_island_info(island_info):

    island_info_path = os.path.join(os.getcwd(), 'data', 'info_islands')
    island = island_info['general_info']['island']
    country = island_info['general_info']['country']

    # Save dictionary
    with open(os.path.join(island_info_path, 'info_{}_{}.data'.format(island, country)), 'wb') as f:
        pickle.dump(island_info, f)

    return island_info

def plot_shoreline_transects(island_info, transect_plot=None, ax=None):
    if ax is None:
        plt.figure()
        ax_plot = plt
    
    else:
        ax_plot = ax

    # Get reference shoreline and transects
    reference_shoreline = island_info['spatial_reference']['reference_shoreline']
    transects = island_info['spatial_reference']['transects']

    # Plot reference shoreline
    ax_plot.plot(reference_shoreline[:, 0], reference_shoreline[:, 1], 'k-')
    
    # Plot transects
    for t in transects:
        if transect_plot is None:
            ax_plot.plot(transects[t][:, 0], transects[t][:, 1], 'r-')
            ax_plot.text(transects[t][-1, 0], transects[t][-1, 1], t, fontsize=20)

        else:
            if t == transect_plot:
                ax_plot.plot(transects[t][:, 0], transects[t][:, 1], 'r-', label='Transect {}'.format(t))

            else:
                ax_plot.plot(transects[t][:, 0], transects[t][:, 1], 'k-', alpha=0.3)

    if transect_plot is not None:
        ax_plot.legend(fontsize=20)
    
    if ax is None:
        ax_plot.xlabel('Longitude', fontsize=20)
        ax_plot.ylabel('Latitude', fontsize=20)

    ax_plot.axis('equal')

    if ax is None:
        plt.show()