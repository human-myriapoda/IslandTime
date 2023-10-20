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
    if argument == 'Faruhulhudhoo':
        date_range = ['2022-10-24', '2022-12-31']
        sat_list = ['S2']
    elif argument == 'Faruhulhedhoo':
        date_range = ['2022-04-02', '2022-12-31']
        sat_list = ['S2']
    elif argument == 'Kodagehutta':
        date_range = ['2020-05-12', '2022-12-31']
        sat_list = ['S2']
    elif argument == 'Kolamaafushi':
        date_range = ['2020-09-30', '2022-12-31']
        sat_list = ['S2']
    elif argument == 'Lhossaa':
        date_range = ['2019-02-12', '2022-12-31']
        sat_list = ['S2']
    else:
        sat_list = ['S2']
        date_range = ['2021-09-09', '2022-12-31']

    # Process satellites in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(
            IslandTime.TimeSeriesCoastSat(argument, 'Maldives', overwrite=True, re_download=True, sat_list=sat_list, date_range=date_range).main()
        )

    island_info = IslandTime.run_all(argument, 'Maldives')
    return f"Processed argument: {argument}"

if __name__ == '__main__':
    arguments = ['Faruhulhudhoo', 'Faruhulhedhoo', 'Kodagehutta', 'Kolamaafushi', 'Lhossaa', 'Maakanaarataa']
    num_cores = 16  # Adjust based on your system capabilities

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_task, arguments)

    for result in results:
        print(result)

'''
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--island", type=str)
args = parser.parse_args()

if args.island != None:
    island_info = IslandTime.run_all(args.island, 'Maldives')

else:
'''