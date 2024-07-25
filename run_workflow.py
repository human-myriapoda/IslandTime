from IslandTime import Workflow, retrieve_island_info
 
for island in ['Gan (Gaafu Dhaalu)']: 
  
    ii = Workflow(island, 'Maldives', run_all=False, execute_analysis=True, execute_segmentation=False, execute_preprocess=False, update_maps=False, overwrite_analysis=True).main()