# :palm_tree: `IslandTime` :palm_tree:
Welcome to `IslandTime`!
This Workflow allows you to extract and analyse satellite images and localised time series for small coral reef islands in Small Island Developing States (SIDS). It uses a combination of multiple existing packages and modules, including `Rbeast` for time series decomposition and `CoastSat` for the extraction of images from Google Earth Engine and pre-processing. 

Note that the Workflow is currently tailored to analyse coral reef islands located in the Maldives, but we are working on expanding the analysis usability to other SIDS.

## How to use from scratch?
Create a `Python 3.11` environment using `requirements.txt` and `environment.yml`. You will need a `Google Earth Engine` account. Note that this environment contains a lot of packages, so it might take a while to create the environment.
Open `workflow_example.ipynb` and input the name and country of the island. Make sure to use a publicly-available name (such as the one used in `OpenStreetMap` or on Google Maps), as this will allow certain actions to be done automatically.

## Structure of the `.data` files


