from mpetools import IslandTime
import argparse
import multiprocessing
import ee

try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()

#arguments = ['Maavaarulu', ]

arguments = ['Gaddhoo'] #['Thinadhoo']#, , 'Maavaarulu', 'Kaludirehaa']#, 'Vaadhoo (Gaafu Dhaalu)',
            #, 'Hoandeddhoo', , ,
            #'Fares Maathodaa', 'Kanduhulhudhoo'] 

def process_task(argument):
    IslandTime.TimeSeriesCoastSat(argument, 'Maldives', overwrite=True, re_download=True, sat_list=['S2'], date_range=['2022-08-20', '2022-12-31']).main()
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
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--island", type=str)
args = parser.parse_args()

if args.island != None:
    island_info = IslandTime.run_all(args.island, 'Maldives')

else:
'''