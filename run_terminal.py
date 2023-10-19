from mpetools import IslandTime
import argparse
import multiprocessing
import ee
import concurrent.futures

try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

def process_task(argument):
    if argument == 'Kuramati':
        date_range = ['2020-07-06', '2022-12-31']
    elif argument == 'Hoandeddhoo':
        date_range = ['2021-02-06', '2022-12-31']
    elif argument == 'Fares Maathodaa':
        date_range = ['2021-07-21', '2022-12-31']
    else:
        date_range = ['2019-12-09', '2022-12-31']

    # Process satellites in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(
            IslandTime.TimeSeriesCoastSat(argument, 'Maldives').main()
        )

    island_info = IslandTime.run_all(argument, 'Maldives')
    return f"Processed argument: {argument}"

if __name__ == '__main__':
    arguments = ['Kuramati', 'Kudhelifadhoo', 'Lifadhoo', 'Kautihulhudhoo-Karuna', 'Hulhedhdhoo', 'Kudhehulhahdhoo']
    num_cores = 16  # Adjust based on your system capabilities

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_task, arguments)

    for result in results:
        print(result)

'''
arguments = ['Vaadhoo (Gaafu Dhaalu)']#,
            #'Hoandeddhoo',
            #'Fares Maathodaa', 'Kanduhulhudhoo']

def process_task(argument):
    if argument == 'Vaadhoo (Gaafu Dhaalu)':
        IslandTime.TimeSeriesCoastSat(argument, 'Maldives', overwrite=True, re_download=True, sat_list=['S2'], date_range=['2019-02-02', '2022-12-31']).main()
    
    elif argument == 'Hoandeddhoo':
        IslandTime.TimeSeriesCoastSat(argument, 'Maldives', overwrite=True, re_download=True, sat_list=['S2'], date_range=['2016-10-20', '2022-12-31']).main()

    elif argument == 'Fares Maathodaa':
        IslandTime.TimeSeriesCoastSat(argument, 'Maldives', overwrite=True, re_download=True, sat_list=['S2'], date_range=['2016-08-11', '2022-12-31']).main()
    
    else:
        IslandTime.TimeSeriesCoastSat(argument, 'Maldives', overwrite=True, re_download=True, sat_list=['S2'], date_range=['2021-03-26', '2022-12-31']).main()
    island_info = IslandTime.run_all(argument, 'Maldives')
    return f"Processed argument: {argument}"

if __name__ == '__main__':
    # Number of CPU cores to use (32 in this case)
    num_cores = 16

    # Create a pool of processes
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Use the pool to map the process_task function to the list of arguments
        results = pool.map(process_task, arguments)

    # Print the results
    for result in results:
        print(result)

        '''

'''
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--island", type=str)
args = parser.parse_args()

if args.island != None:
    island_info = IslandTime.run_all(args.island, 'Maldives')

else:
'''