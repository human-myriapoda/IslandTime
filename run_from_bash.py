from mpetools import IslandTime
import multiprocessing
import concurrent.futures
import numpy as np
import os
import datetime

def process_task_total(argument):

    # Process satellites in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(
            IslandTime.TimeSeriesCoastSat(argument, 'Maldives').main()
        )

    _ = IslandTime.run_all(argument, 'Maldives')
    return f"Processed argument: {argument}"

def process_task_partially(arg, sat_list, date_range):

    # Process satellites in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(
            IslandTime.TimeSeriesCoastSat(arg, 'Maldives', re_download=True, overwrite=True, sat_list=sat_list, date_range=date_range).main()
        )

    _ = IslandTime.run_all(arg, 'Maldives')
    return f"Processed argument: {arg}"

def get_last_ms_date(path):
    ms_files = sorted(os.listdir(path))
    if ms_files:
        last_ms_file = ms_files[-1]
        i = -1
        while not last_ms_file.endswith('_ms.tif'):
            last_ms_file = ms_files[i]
            i -= 1
        last_ms_file = last_ms_file.split('.')[0]
        date_last_file = last_ms_file.split('-')[:3]
        return datetime.datetime(int(date_last_file[0]), int(date_last_file[1]), int(date_last_file[2])+1)
    return None

def check_and_update_dict(island, country, satellite, dict_batch):
    folder_path = os.path.join(os.getcwd(), 'data', 'coastsat_data', '{}_{}'.format(island, country), satellite, 'ms')
    last_date = get_last_ms_date(folder_path)

    if last_date and last_date > datetime.datetime(2022, 11, 10):
        return dict_batch
    
    dict_batch[island] = {
        'sat_list': [satellite],
        'date_range': ['{}-{}-{}'.format(last_date.year, last_date.month, last_date.day) if last_date else '2010-01-01', '2022-12-31']
    }

    return dict_batch

# Path to `info_islands` folder
path_to_data = os.path.join(os.getcwd(), 'data', 'info_islands')

# Create empty lists
downloaded, partially_downloaded, not_downloaded = [], [], []

# Loop in every file
for file in os.listdir(path_to_data):
    island = file.split('_')[1]
    country = file.split('_')[2].split('.')[0]
    island_info = IslandTime.retrieve_island_info(island, country, verbose=False)

    # Check if island has been downloaded
    if 'timeseries_coastsat' in island_info.keys():
        if 'timeseries' in island_info['timeseries_coastsat'].keys():
            if all([sat in os.listdir(os.path.join(os.getcwd(), 'data', 'coastsat_data', '{}_{}'.format(island, country))) for sat in ['S2', 'L8', 'L9']]):
                downloaded.append(island)
            else:
                partially_downloaded.append(island)
        
        else:
            if os.path.exists(os.path.join(os.getcwd(), 'data', 'coastsat_data', island+'_'+country)):
                partially_downloaded.append(island)
    
    else:
        if os.path.exists(os.path.join(os.getcwd(), 'data', 'coastsat_data', island+'_'+country)):
            partially_downloaded.append(island)
        
        else:
            not_downloaded.append(island)

# Create batch of islands to download
batch_size = 8
dict_batch = {}
country = 'Maldives'
batch_partially_downloaded = False

# Prioritise islands that have been partially downloaded
if len(partially_downloaded) > 1:
    batch_partially_downloaded = True

    if len(partially_downloaded) < batch_size:
        batch = partially_downloaded
    else:
        batch = np.random.choice(partially_downloaded, size=batch_size, replace=False)

    # Define downloading parameters
    for arg in batch:
        dict_batch[arg] = {}

        if 'L8' in os.listdir(os.path.join(os.getcwd(), 'data', 'coastsat_data', '{}_{}'.format(arg, country))):
            dict_batch = check_and_update_dict(arg, country, 'L8', dict_batch)
            if dict_batch[arg]:
                continue
        
        if not 'L8' in os.listdir(os.path.join(os.getcwd(), 'data', 'coastsat_data', '{}_{}'.format(arg, country))):
            dict_batch[arg] = {
                'sat_list': ['L8'],
                'date_range': ['2010-01-01', '2022-12-31']
            }
            continue

        if 'L9' in os.listdir(os.path.join(os.getcwd(), 'data', 'coastsat_data', '{}_{}'.format(arg, country))):
            dict_batch = check_and_update_dict(arg, country, 'L9', dict_batch)
            if dict_batch[arg]:
                continue
        
        if not 'L9' in os.listdir(os.path.join(os.getcwd(), 'data', 'coastsat_data', '{}_{}'.format(arg, country))):
            dict_batch[arg] = {
                'sat_list': ['L9'],
                'date_range': ['2010-01-01', '2022-12-31']
            }
            continue

        if 'S2' in os.listdir(os.path.join(os.getcwd(), 'data', 'coastsat_data', '{}_{}'.format(arg, country))):
            dict_batch = check_and_update_dict(arg, country, 'S2', dict_batch)
            if dict_batch[arg]:
                continue
        
        if not 'S2' in os.listdir(os.path.join(os.getcwd(), 'data', 'coastsat_data', '{}_{}'.format(arg, country))):
            dict_batch[arg] = {
                'sat_list': ['S2'],
                'date_range': ['2010-01-01', '2022-12-31']
            }
            continue 
    
    # Create tuple of arguments
    tuple_args = []
    for arg_dict in dict_batch.keys():
        if dict_batch[arg_dict]:
            tuple_args.append((arg_dict, dict_batch[arg_dict]['sat_list'], dict_batch[arg_dict]['date_range']))

# Download islands that have not been downloaded
else:
    if len(not_downloaded) < batch_size:
        batch = not_downloaded
    else:
        batch = np.random.choice(not_downloaded, size=batch_size, replace=False)

# Main function
if __name__ == '__main__':
    num_cores = 16  # Adjust based on your system capabilities

    if batch_partially_downloaded:
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.starmap(process_task_partially, tuple_args)

    else:
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = pool.map(process_task_total, batch)

    for result in results:
        print(result)


