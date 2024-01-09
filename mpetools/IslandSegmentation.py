"""
This module contains tools to segment islands from satellite images. It outputs a dictionary with the polygons of the island as well as coastline position time series data.
TODO: Add docstrings
TODO: Loop of bands within the segmentation function
TODO: clean functions
Author: Myriam Prasow-Emond, Department of Earth Science and Engineering, Imperial College London
"""

# Import modules
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_multiotsu
from skimage import measure
import scipy.signal
import scipy.stats
import shapely
from celluloid import Camera
import geopandas as gpd
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import pandas as pd
import skimage.transform as transform
import pyproj
from matplotlib.widgets import Button

####################################
from tqdm import tqdm
import pickle
import os
from coastsatmaster.coastsat import SDS_preprocess, SDS_tools

class Segmentation:
    def __init__(self, island_info, list_sat=['S2', 'L8', 'L9']):
        self.island_info = island_info
        self.list_sat = list_sat

        # Extract relevant information from island_info
        self.island = self.island_info['general_info']['island']
        self.country = self.island_info['general_info']['country']
        self.latitude = self.island_info['general_info']['latitude']
        self.longitude = self.island_info['general_info']['longitude']
        self.settings_LS = self.island_info['timeseries_coastsat']['settings']
        self.reference_shoreline = self.island_info['timeseries_coastsat']['reference_shoreline']
    
    def coregistration_PSS(self, visual_inspection=True):
        pass

    def loop_through_files(self):
        pass

    def main(self):

        print('\n-------------------------------------------------------------------')
        print('Segmentation of {}, {}'.format(self.island, self.country))
        print('-------------------------------------------------------------------\n')

        # Load metadata (CoastSat)
        metadata = pickle.load(open(os.path.join(os.getcwd(), 'data', 'satellite_imagery', '{}_{}'.format(self.island, self.country), '{}_{}_metadata.pkl'.format(self.island, self.country)), 'rb'))

        # Loop through list of satellites
        for sat in self.list_sat:
            print('SATELLITE: {}'.format(sat))

            if sat == 'PSS':
                # Define path for PSS images
                filepath = os.path.join(os.getcwd(), 'data', 'satellite_imagery',)

            else:
                # Empty metadata for this satellite (taken from CoastSat)
                if metadata[sat]['filenames'] == []:
                    continue

                # File path and names from CoastSat folder
                filepath = SDS_tools.get_filepath(self.settings_LS['inputs'], sat)
                filenames = metadata[sat]['filenames']
        

        # 1. Extract polygons and save them in a dictionary

        # 2. Best polygons (user)

        # 3. Time series from best polygons
        
        if 'PSS' in self.list_sat:
            pass
            #self.coregistration_PSS()

#######################################################################################################

def plot_polygons(polygons, ax):

    # Default matplotlib colors list
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 1500
    
    # Sort polygons by area
    polygons_sorted_area = sorted(polygons, key=lambda poly: poly.area)

    # Loop over polygons
    for idx_polygon, polygon in enumerate(polygons_sorted_area):
        # If smallest polygon, no difference
        if idx_polygon == 0:
            polygon_difference = polygon

        # If not smallest polygon, difference with all smaller polygons
        else:
            polygon_difference = polygon
            # Loop over smaller polygons
            for idx_smaller_polygon in reversed(range(idx_polygon)):
                # Difference with smaller polygon
                polygon_difference = polygon_difference.difference(polygons_sorted_area[idx_smaller_polygon])
        
        # Plot differenced polygon
        gpd.GeoSeries(polygon_difference).plot(ax=ax, alpha=0.4, color=colors[idx_polygon])

def combine_geometry_extent(contour, img, distance_threshold=30, debug=False):

    # Expand image array with nan values for extent contouring
    expanded_array = np.full((img.shape[0]+2, img.shape[1]+2), np.nan)
    expanded_array[1:img.shape[0]+1, 1:img.shape[1]+1] = img

    # Create a mask with nan values
    nan_mask = np.isnan(expanded_array)

    # Find contour of extent (boundary between nan and non-nan values)
    extent = measure.find_contours(nan_mask, 0.5)

    # Find largest contour
    if len(extent) > 0:
        extent = extent[np.argmax([len(e) for e in extent])]

    # Create shapely objects
    polygon_extent = shapely.geometry.Polygon(extent-1) # remove 1 to account for the expansion
    linestring_extent = shapely.geometry.LineString(polygon_extent.exterior.coords)
    linestring_contour = shapely.geometry.LineString(contour)

    # First and last point of the contour
    first_point_contour = shapely.geometry.Point(linestring_contour.coords[0])
    last_point_contour = shapely.geometry.Point(linestring_contour.coords[-1])

    # Find the closest point on the extent to the first and last point of the contour
    list_distance_first_point = [first_point_contour.distance(shapely.geometry.Point(point_l_extent)) for point_l_extent in linestring_extent.coords]
    corresponding_first_point = linestring_extent.coords[int(np.argmin(list_distance_first_point))]

    list_distance_last_point = [last_point_contour.distance(shapely.geometry.Point(point_l_extent)) for point_l_extent in linestring_extent.coords]
    corresponding_last_point = linestring_extent.coords[int(np.argmin(list_distance_last_point))]

    # If contour and extent are close, combine them
    if min(list_distance_first_point) < distance_threshold and min(list_distance_last_point) < distance_threshold:

        # Create shapely objects
        point1 = shapely.geometry.Point(corresponding_first_point)
        point2 = shapely.geometry.Point(corresponding_last_point)

        # Find the indices of the points on the line
        index1 = None
        index2 = None

        for idx_coord, coord in enumerate(linestring_extent.coords):

            if shapely.geometry.Point(coord).equals(point1):
                index1 = idx_coord
            elif shapely.geometry.Point(coord).equals(point2):
                index2 = idx_coord

        if index1 is not None and index2 is not None:
            
            if index1 > index2:
            # Extract the segment between the two points
                segment1 = shapely.geometry.LineString(linestring_extent.coords[index2:(index1+1)])
                
                if debug:
                    fig, ax = plt.subplots()
                    gpd.GeoSeries(shapely.geometry.LineString(contour)).plot(ax=ax, color='g')
                    gpd.GeoSeries(segment1).plot(ax=ax, color='r')
                    gpd.GeoSeries(linestring_extent.difference(segment1)).plot(ax=ax, color='b')
                    plt.show()

                try:
                    segment2 = shapely.ops.linemerge(linestring_extent.difference(segment1))
                except:
                    return contour

            else:
                segment1 = shapely.geometry.LineString(linestring_extent.coords[index1:(index2+1)])

                if debug:
                    fig, ax = plt.subplots()
                    gpd.GeoSeries(shapely.geometry.LineString(contour)).plot(ax=ax, color='g')
                    gpd.GeoSeries(segment1).plot(ax=ax, color='r')
                    gpd.GeoSeries(linestring_extent.difference(segment1)).plot(ax=ax, color='b')
                    plt.show()

                try:
                    segment2 = shapely.ops.linemerge(linestring_extent.difference(segment1))
                except:
                    return contour

        else:
            return contour
    
        # Combine the contour and the segment
        possible_polygons = np.array([shapely.geometry.Polygon(list(segment1.coords) + list(linestring_contour.coords)),
                        shapely.geometry.Polygon(list(reversed(list(segment1.coords))) + list(linestring_contour.coords)),
                        shapely.geometry.Polygon(list(segment1.coords) + list(reversed(list(linestring_contour.coords)))),
                        shapely.geometry.Polygon(list(reversed(list(segment1.coords))) + list(reversed(list(linestring_contour.coords)))),
                        shapely.geometry.Polygon(list(segment2.coords) + list(linestring_contour.coords)),
                        shapely.geometry.Polygon(list(reversed(list(segment2.coords))) + list(linestring_contour.coords)),
                        shapely.geometry.Polygon(list(segment2.coords) + list(reversed(list(linestring_contour.coords)))),
                        shapely.geometry.Polygon(list(reversed(list(segment2.coords))) + list(reversed(list(linestring_contour.coords))))])


        # List of valid polygons
        bool_valid_polygons = [poly.is_valid for poly in possible_polygons]

        # Select valid polygons
        valid_polygons = possible_polygons[bool_valid_polygons]

        # Calculate area of valid polygons
        area_poly = [poly.area for poly in valid_polygons]

        # Check which polygon is valid
        for poly in valid_polygons:
            if poly.is_valid and poly.area == min(area_poly):
                return np.array(poly.exterior.coords)
            
        return contour

    # Return contour
    else:
        return contour

def combine_geometry_close_endpoints(line1, line2, distance_threshold=10):

    # List of endpoints for both lines
    endpoints = [
        shapely.geometry.Point(line1.coords[0]),
        shapely.geometry.Point(line1.coords[-1]),
        shapely.geometry.Point(line2.coords[0]),
        shapely.geometry.Point(line2.coords[-1])
    ]

    # List of distances between endpoints
    distances = [endpoints[0].distance(endpoints[2]),
                 endpoints[0].distance(endpoints[3]),
                 endpoints[1].distance(endpoints[2]),
                 endpoints[1].distance(endpoints[3])]

    # If one of the endpoints is close to another endpoint, combine the lines
    if min(distances) < distance_threshold:
        combined_lines = [
            shapely.geometry.LineString(list(line1.coords) + list(line2.coords)),
            shapely.geometry.LineString(list(line1.coords) + list(reversed(line2.coords))),
            shapely.geometry.LineString(list(line1.coords) + list(reversed(line2.coords))),
            shapely.geometry.LineString(list(reversed(line1.coords)) + list(reversed(line2.coords)))
        ]

        # Pick the shortest line
        combined_line = combined_lines[np.argmin([line.length for line in combined_lines])]

        return combined_line
    
    else:
        return None

def close_contours(contours, distance_threshold=10):

    for c in range(len(contours)):
        ls = shapely.geometry.LineString(contours[c])
        p1 = shapely.geometry.Point(ls.coords[0])
        p2 = shapely.geometry.Point(ls.coords[-1])

        if p1.distance(p2) < distance_threshold:
            contours[c] = np.concatenate([contours[c], [contours[c][0]]], axis=0)
    
    return contours

def combine_contours(contours, img, verbose, min_len=100, plot_int_debug=False):

    # Close almost closed contours
    contours = close_contours(contours)

    # Make list of open and closed contours
    contours_open = [c for c in contours if (len(c) > min_len) and not (shapely.geometry.LineString(c).is_closed)]
    contours_closed = [c for c in contours if (len(c) > min_len) and (shapely.geometry.LineString(c).is_closed)]

    # If all contours are closed, return
    if contours_open == []:
        return contours

    # Optional plotting for debugging
    if plot_int_debug:
        plt.figure()
        plt.title('Contours before combining close endpoints')
        for i in range(len(contours_open)):
            plt.plot(contours_open[i][:, 1], contours_open[i][:, 0], label=i)
        plt.legend()
        plt.axis('equal')
        plt.show()
    
    to_exclude = []
    
    # Combine contours with close endpoints
    for i in range(len(contours_open)):
        if i not in to_exclude:
            # LineString of the contour
            line1 = shapely.geometry.LineString(contours_open[i])

            for j in range(len(contours_open)):
                if i != j and j not in to_exclude:
                    line2 = shapely.geometry.LineString(contours_open[j])

                    lines = combine_geometry_close_endpoints(line1, line2)
                    
                    if lines is not None:
                        if verbose:
                            print('contour {} is combined with contour {}'.format(i, j))
                        contours_open[i] = np.array(lines.coords)
                        contours_open[j] = np.array(lines.coords)
                        # update contours in to_exclude
                        for k in to_exclude:
                            contours_open[k] = np.array(lines.coords)
                        to_exclude.append(j)

    # Remove duplicates
    unique_contours = []
    for arr in contours_open:
        if not any(np.array_equal(arr, unique_arr) for unique_arr in unique_contours):
            unique_contours.append(arr)
    
    # Update contours_open
    contours_open = unique_contours
    
    # Optional plotting for debugging
    if plot_int_debug:
        plt.figure()
        plt.title('Contours after combining close endpoints')
        for i in range(len(contours_open)):
            plt.plot(contours_open[i][:, 1], contours_open[i][:, 0], label=i)
        plt.legend()
        plt.axis('equal')
        plt.show()

    # Combine contours with extent
    for con in range(len(contours_open)):
        contours_open[con] = combine_geometry_extent(contours_open[con], img)

    # Optional plotting for debugging
    if plot_int_debug:
        plt.figure()
        for con in contours_open:
            plt.plot(con[:, 1], con[:, 0], label=i)
        plt.legend()
        plt.axis('equal')
        plt.title('Contours after adding extent points')
        plt.show()

    # Add closed contours
    for con in contours_closed:
        contours_open.append(con)

    return contours_open

def histogram_pixel_values(img_no_nan, ax, plot_results, method='smoothing', sigma=10.):

    # Create linspace of pixel values between min and max
    X = np.linspace(min(img_no_nan), max(img_no_nan), 1000)

    # Create histogram distribution for smoothing
    if method == 'kde':
        kernel = scipy.stats.gaussian_kde(img_no_nan.flatten())
        histogram_distribution = kernel(X)
    
    elif method == 'smoothing':
        hist = np.histogram(img_no_nan.flatten(), bins=int(np.sqrt(len(img_no_nan.flatten()))))
        hist_dist = scipy.stats.rv_histogram(hist)
        smoothed_signal = scipy.ndimage.gaussian_filter(hist_dist.pdf(X), sigma=sigma)
        histogram_distribution = smoothed_signal

    if plot_results:
        # Plotting
        ax.plot(X, histogram_distribution, color='r', label=method)
        ax.hist(img_no_nan.flatten(), bins=int(np.sqrt(len(img_no_nan.flatten()))), density=True, color='k', alpha=0.3, label='histogram')

    # Find peaks and minima
    peaks, _ = scipy.signal.find_peaks(histogram_distribution, height=0.1)
    minima = scipy.signal.argrelextrema(histogram_distribution, np.less)[0]

    if plot_results:
        # Plot peaks and minima
        for idx_peak, peak in enumerate(peaks):
            if idx_peak == 0:
                ax.axvline(X[peak], color='b', label='peaks')
            else:
                ax.axvline(X[peak], color='b')

        for idx_mini, mini in enumerate(minima): 
            if idx_mini == 0:
                ax.axvline(X[mini], color='g', label='minima')
            else:
                ax.axvline(X[mini], color='g')

    # Return X values at peaks and minima
    X_minima = X[minima]
    X_peaks = X[peaks]

    return X_minima, X_peaks, X, histogram_distribution

def create_animation(polygons_dict, rgb, img, X, histogram_distribution, label):
    
    # Create figure
    figanim, (axanim1, axanim2) = plt.subplots(1, 2)

    # Initialisation camera
    camera = Camera(figanim)
    
    # All thresholds
    thresholds = [polygons_dict[key]['threshold'] for key in polygons_dict.keys()]

    # Loop over thresholds
    for key in polygons_dict.keys():
        
        axanim1.plot(X, histogram_distribution, color='k', label='histogram')
        axanim1.hist(img.flatten(), bins=int(np.sqrt(len(img.flatten()))), density=True, color='k', alpha=0.3, label='histogram')
        axanim2.imshow(rgb)
        axanim2.set_title('RGB image')
        axanim1.set_title('pixel value histogram: {}'.format(label))

        for threshold in thresholds:
            if threshold == polygons_dict[key]['threshold']:
                axanim1.axvline(threshold, color='orange')
            else:
                axanim1.axvline(threshold, color='orange', alpha=0.2)
        
        for polygon in polygons_dict[key]['polygons']:
            gpd.GeoSeries(polygon).plot(ax=axanim2, alpha=0.4, color='r')

        camera.snap()

    animation = camera.animate()
    animation.save('{}.gif'.format(label), writer='Pillow', fps=1)
    
def segmentation(img, rgb, label, segmented_image, cmap, classes_to_consider=None, plot_results=False, animation=False, verbose=False, find_polygons=True):
    
    if verbose:
        print('\n-------------------------------------------------------------------')
        print('Segmentation of {} image'.format(label))
        print('-------------------------------------------------------------------\n')

    if plot_results:
        # Subplot initialisation
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
    else:
        fig, ax1, ax2, ax3, ax4 = None, None, None, None, None

    # Replace 0 values with nan
    img = np.where(img == 0., np.nan, img)
    img_no_nan = img[~np.isnan(img)]

    # Otsu threshold (2 classes)
    f_otsu_two_classes = threshold_multiotsu(img_no_nan, classes=2)

    # Otsu threshold between classes of interest
    '''
    mask_class1 = np.where(segmented_image == classes_to_consider[0], 1, 0)
    mask_class2 = np.where(segmented_image == classes_to_consider[1], 1, 0)
    total_mask = mask_class1 + mask_class2
    img_two_classes = np.where(total_mask == 1, img, np.nan)
    img_two_classes_no_nan = img_two_classes[~np.isnan(img_two_classes)]
    f_otsu_two_classes = threshold_multiotsu(img_two_classes_no_nan, classes=2)
    '''
    # Histogram of pixel values
    X_minima, X_peaks, X, histogram_distribution = histogram_pixel_values(img_no_nan, ax=ax3, plot_results=plot_results, method='smoothing')

    #if plot_results:
    #    associate_peaks_to_classification(img, segmented_image, label, X, histogram_distribution, X_peaks, cmap)

    # Create empty list of polygons
    polygons = []
    polygons_dict = {}

    # List of thresholds
    if find_polygons:
        # Threshold for optimisation (0.30 around the Otsu threshold)
        thresholds = np.linspace(f_otsu_two_classes[0]-0.30, f_otsu_two_classes[0]+0.30, 30)
    
    else:
        thresholds = X_minima

    # Loop over thresholds (minima of histogram)
    for idx_minimum, minimum in enumerate(list(reversed(thresholds))):

        # Find contours with skimage.measure.find_contours (Marching Squares algorithm)
        contours = measure.find_contours(img, minimum, fully_connected='high', positive_orientation='high')

        # Get Otsu contour if label is NIR, NDWI and NDVI
        if label in ['NIR', 'NDWI', 'NDVI', 'MNDWI', 'red', 'green', 'blue']:
            contours_otsu = measure.find_contours(img, f_otsu_two_classes, fully_connected='high', positive_orientation='high')

            if find_polygons:
                # Remove small contours
                contours_otsu = [c for c in contours_otsu if len(c) > 200]

            contours_combined_otsu = combine_contours(contours_otsu, img, verbose=verbose)
            polygons_dict['otsu'] = {'label': label,
                                     'threshold': f_otsu_two_classes[0],
                                     'polygons': [shapely.geometry.Polygon(np.flip(c)) for c in contours_combined_otsu]}

        # Combine contours
        if find_polygons:
            # Remove small contours
            contours = [c for c in contours if len(c) > 200]

        contours_combined = combine_contours(contours, img, verbose=verbose)

        # Add polygons to dictionary (with threshold as key)
        polygons_dict[idx_minimum] = {'label': label,
                                      'threshold': minimum,
                                      'polygons': [shapely.geometry.Polygon(np.flip(c)) for c in contours_combined]}
        
        # Add polygons to list
        for idx_c, c in enumerate(contours_combined):
            if shapely.geometry.Polygon(np.flip(c)).is_valid:
                polygons.append(shapely.geometry.Polygon(np.flip(c)))

        if plot_results:
            # Plot contours
            for cc in contours_combined:
                ax1.plot(cc[:, 1], cc[:, 0])
                #ax2.plot(cc[:, 1], cc[:, 0])
    
    if plot_results:
        # Plot polygons
        plot_polygons(polygons, ax=ax4)

        # Additional plotting parameters    
        ax1.imshow(rgb)
        ax1.set_title('RGB image and contours {}'.format(label))
        ax2.imshow(img)
        ax2.set_title('{} image'.format(label))
        ax3.axvline(f_otsu_two_classes, color='orange', label='Otsu threshold')
        ax3.legend()
        ax3.set_title('pixel value histogram: {}'.format(label))
        ax4.set_title('polygons: {}'.format(label))
        ax4.imshow(rgb)
        plt.show()     

    if animation:
        create_animation(polygons_dict, rgb, img, X, histogram_distribution, label)

    return polygons_dict, X_peaks, X, histogram_distribution

def unsupervised_classification(stacked_array, n_clusters, plot_classification=False):

    imputer = SimpleImputer(strategy='mean')

    for i in range(stacked_array.shape[2]):
        stacked_array[:, :, i] = imputer.fit_transform(stacked_array[:, :, i])

    # Reshape the image to a 2D array of pixels (rows) by RGB values (columns)
    image_shape = stacked_array.shape
    image_2d = stacked_array.reshape((-1, image_shape[2]))

    # Fit the k-means model to the image data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(image_2d)

    # Get the labels assigned to each pixel
    labels = kmeans.labels_

    # Reshape the labels back to the shape of the original image
    segmented_image = labels.reshape(image_shape[0], image_shape[1])

    if plot_classification:

        cmap = ListedColormap(['turquoise', 'coral', 'green', 'blueviolet', 'gold', 'silver', 'hotpink'][:n_clusters+1])

        # Display the segmented image
        plt.figure()
        cbar = plt.imshow(segmented_image, cmap=cmap) 
        plt.colorbar(cbar)
        plt.axis('off')
        plt.show(block=False)
    else:
        cmap = None
    
    # TODO: Re-assign labels for consistency

    return segmented_image, cmap

def associate_peaks_to_classification(img, segmented_image, label_image, X, kde, peaks, cmap):

    # Classification labels
    labels = np.unique(segmented_image.flatten())

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Plot histogram, KDE and peaks
    ax1.set_title('Histogram and KDE of {} image'.format(label_image))
    ax1.hist(img.flatten(), bins=int(np.sqrt(len(img.flatten()))), alpha=0.2, color='k', density=True)
    ax1.plot(X, kde, color='r', label='KDE')
    for peak in peaks:
        ax1.axvline(peak, color='b', label='peaks')

    # Plot histogram for each label
    for label in np.unique(labels):
        mask = np.where(segmented_image == label, 1, 0)
        masked_nir = np.where(mask == 1, img, np.nan)
        ax1.hist(masked_nir.flatten(), bins=1000, label=label, alpha=0.5, color=cmap.colors[label], density=True)
    
    # Plot segmented image for comparison
    ax2.set_title('Segmented image')
    cbar = ax2.imshow(segmented_image, cmap=cmap)
    plt.colorbar(cbar)

    plt.show()

def return_polygons_coi(polygons_dict, center_of_image):
    # Empty lists
    polygons_coi = []
    thresholds_coi = []

    # Loop over thresholds
    for key in polygons_dict.keys():
        # If polygons are found for this threshold
        if len(polygons_dict[key]['polygons']) > 0:
            # Find polygons that contain the center of the image
            idx_coi = [polygons_dict[key]['polygons'][i].contains(center_of_image) for i in range(len(polygons_dict[key]['polygons']))]
            polygons_c = np.array(polygons_dict[key]['polygons'])[idx_coi]

            # If polygons are found
            if len(polygons_c) > 0:
                # If only one polygon is found, return this polygon
                if len(polygons_c) == 1:
                    if polygons_c[0].is_valid:
                        poly_coi = polygons_c[0]
                    
                    else:
                        continue

                # If multiple polygons are found, return the largest polygon
                else:
                    poly = polygons_c[np.argmax([polygon.area for polygon in polygons_c])]
                    if poly.is_valid:
                        poly_coi = poly
                    
                    else:
                        continue

                # Append lists
                polygons_coi.append(poly_coi)
                thresholds_coi.append(polygons_dict[key]['threshold'])
    
    return np.array(polygons_coi), np.array(thresholds_coi)

def find_optimal_polygon(polygons_dicts, center_of_image, rgb, plot_results=True, key=None, metrics='length'):
    
    dict_NIR = polygons_dicts[0]
    dict_NDVI = polygons_dicts[1]
    dict_NDWI = polygons_dicts[2]

    poly_NDWI, threshold_NDWI = return_polygons_coi(dict_NDWI, center_of_image)
    poly_NDVI, threshold_NDVI = return_polygons_coi(dict_NDVI, center_of_image)
    poly_NIR, threshold_NIR = return_polygons_coi(dict_NIR, center_of_image)

    # Create a list to store dictionaries of results
    results_list = []

    if len(poly_NDWI) == 0 or len(poly_NDVI) == 0 or len(poly_NIR) == 0:
        return None

    # Create a meshgrid of indices
    i, j, k = np.meshgrid(np.arange(len(poly_NDWI)), np.arange(len(poly_NDVI)), np.arange(len(poly_NIR)))

    # Flatten the indices
    indices_flat = np.column_stack((i.flatten(), j.flatten(), k.flatten()))

    # Iterate through the indices
    for index in indices_flat:
        i, j, k = index
        polygons_to_test = [poly_NDWI[i], poly_NDVI[j], poly_NIR[k]]
        standard_dev_area = np.std([polygon.area for polygon in polygons_to_test])
        standard_dev_length = np.std([polygon.exterior.length for polygon in polygons_to_test])

        if metrics == 'area':
            metrics_std = standard_dev_area
        
        elif metrics == 'length':
            metrics_std = standard_dev_length

        # Append the results to the list
        results_list.append({
            'Area Difference': metrics_std,
            'Threshold MNDWI': threshold_NDWI[i],
            'Threshold NDVI': threshold_NDVI[j],
            'Threshold NIR': threshold_NIR[k]
        })

    # Create a DataFrame from the list of dictionaries
    df_results = pd.DataFrame(results_list)

    # Find the minimum area difference (using argsort for debugging purposes)
    min_idx = df_results['Area Difference'].argsort()[0]

    # Polygons and thresholds with the minimum area difference
    poly_NDWI_opt = poly_NDWI[np.argwhere(threshold_NDWI == df_results.iloc[min_idx]['Threshold MNDWI'])[0][0]]
    poly_NDVI_opt = poly_NDVI[np.argwhere(threshold_NDVI == df_results.iloc[min_idx]['Threshold NDVI'])[0][0]]
    poly_NIR_opt = poly_NIR[np.argwhere(threshold_NIR == df_results.iloc[min_idx]['Threshold NIR'])[0][0]]

    # Otsu threshold polygones
    if len(dict_NDWI['otsu']['polygons']) > 0:
        poly_NDWI_otsu = dict_NDWI['otsu']['polygons'][np.argmax([polygon.area for polygon in dict_NDWI['otsu']['polygons']])]
    else:
        poly_NDWI_otsu = None
    
    if len(dict_NDVI['otsu']['polygons']) > 0:
        poly_NDVI_otsu = dict_NDVI['otsu']['polygons'][np.argmax([polygon.area for polygon in dict_NDVI['otsu']['polygons']])]
    else:
        poly_NDVI_otsu = None
    
    if len(dict_NIR['otsu']['polygons']) > 0:
        poly_NIR_otsu = dict_NIR['otsu']['polygons'][np.argmax([polygon.area for polygon in dict_NIR['otsu']['polygons']])]
    else:
        poly_NIR_otsu = None

    if key is None:
        # List of optimal polygons + NIR Otsu
        polygons_opt_otsu = np.array([poly_NDWI_opt, poly_NDVI_opt, poly_NIR_opt, poly_NIR_otsu])

        # Remove None
        polygons_opt_otsu = polygons_opt_otsu[np.argwhere(polygons_opt_otsu != None).flatten()]

        # List of all polygons for plotting
        polygons_opt_otsu_plot = np.array([poly_NDWI_opt, poly_NDVI_opt, poly_NIR_opt, poly_NIR_otsu, poly_NDWI_otsu, poly_NDVI_otsu])

        # Remove None
        polygons_opt_otsu_plot = polygons_opt_otsu_plot[np.argwhere(polygons_opt_otsu_plot != None).flatten()]
        
        # Remove outlier polygons
        polygons_opt_otsu = polygons_opt_otsu[np.argwhere(abs(scipy.stats.zscore([polygon.area for polygon in polygons_opt_otsu])) < 1.5).flatten()]

        if plot_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
            ax1.imshow(rgb)
            colors = ['r', 'g', 'b', 'y', 'm', 'c']
            labels = ['MNDWI', 'NDVI', 'NIR', 'NIR Otsu', 'MNDWI Otsu', 'NDVI Otsu']
            for i, poly in enumerate(polygons_opt_otsu_plot):
                gpd.GeoSeries(poly.exterior).plot(ax=ax1, color=colors[i], label=labels[i])
            ax1.legend()

        # Polygon with the smallest area
        polygon_smallest_area = polygons_opt_otsu[np.argsort([polygon.area for polygon in polygons_opt_otsu])[0]]

        # List without the smallest polygon
        polygons_opt_otsu_w = polygons_opt_otsu[np.argsort([polygon.area for polygon in polygons_opt_otsu])[1:]]

        # Empty list
        point_mean_t = []

        # Find average polygon
        for x in polygon_smallest_area.exterior.coords:
            # Create shapely object and empty lists
            pt = shapely.geometry.Point(x)
            corresponding_point = [x]

            # Loop to find the closest point in the polygon
            for poly in polygons_opt_otsu_w:
                poly_coords = np.array(poly.exterior.coords)
                list_distance_point = np.linalg.norm(poly_coords - pt.coords, axis=1)
                min_distance_index = np.argmin(list_distance_point)
                corresponding_point.append(poly_coords[min_distance_index])

            corresponding_point = np.array(corresponding_point)
            point_mean = shapely.geometry.Point(np.mean(corresponding_point[:, 0]), np.mean(corresponding_point[:, 1]))
            point_mean_t.append(point_mean)

        polygon_mean = shapely.geometry.Polygon(point_mean_t)

        if plot_results:
            ax2.imshow(rgb)
            gpd.GeoSeries(polygon_mean.exterior).plot(ax=ax2, alpha=0.4, color='r')
            plt.show(block=False)
        
        # Build dictionary with all results
        res_opt = {'polygon_opt': polygon_mean,
                'polygon_NDWI_opt': poly_NDWI_opt,
                'polygon_NDVI_opt': poly_NDVI_opt,
                'polygon_NIR_opt': poly_NIR_opt,
                'polygon_NIR_otsu': poly_NIR_otsu,
                'polygon_NDWI_otsu': poly_NDWI_otsu,
                'polygon_NDVI_otsu': poly_NDVI_otsu}
        
        return res_opt

    elif key == 'n':
        return poly_NIR_opt
    
    elif key == 'v':
        return poly_NDVI_opt
    
    elif key == 'w':
        return poly_NDWI_opt
    
    elif key == 'no':
        return poly_NIR_otsu
    
    elif key == 'vo':
        return poly_NDVI_otsu
    
    elif key == 'wo':
        return poly_NDWI_otsu

def time_series_from_polygons(island_info, dict_poly, dict_rgb, dict_georef,  dict_image_epsg, dict_best_polygons):

    # Extract reference shoreline and transects from island_info
    reference_shoreline = island_info['spatial_reference']['reference_shoreline']
    transects = island_info['spatial_reference']['transects']

    # Empty dictionary
    dict_time_series = {}

    # Dictionary with corresponding names
    dict_names = {'NIR': 'polygon_NIR_opt',
                    'NDVI': 'polygon_NDVI_opt',
                    'NDWI': 'polygon_NDWI_opt',
                    'NIR Otsu': 'polygon_NIR_otsu',
                    'NDVI Otsu': 'polygon_NDVI_otsu',
                    'NDWI Otsu': 'polygon_NDWI_otsu',
                    'Optimal': 'polygon_opt'}

    # Transform polygons to image crs and calculate position on transect
    fig, ax = plt.subplots()
    for key in dict_best_polygons.keys():
        # Find corresponding polygon
        key_to_name = dict_best_polygons[key]
        polygon = dict_poly[key][dict_names[key_to_name]]

        # Get coordinates of polygon (exterior)
        x_poly, y_poly  = polygon.exterior.coords.xy

        # Get image epsg
        image_epsg = dict_image_epsg[key]

        # Define projection
        src_crs = pyproj.CRS('EPSG:3857')
        tgt_crs = pyproj.CRS('EPSG:{}'.format(image_epsg))

        # Create transformer
        transformer = pyproj.Transformer.from_crs(src_crs, tgt_crs, always_xy=True)

        # Transform reference shoreline
        shoreline_x_transformed, shoreline_y_transformed = transformer.transform(reference_shoreline[:, 0], reference_shoreline[:, 1])
        shoreline_image_crs = np.column_stack((shoreline_x_transformed, shoreline_y_transformed))
        polygon_shoreline_image_crs = shapely.geometry.Polygon(shoreline_image_crs)

        # Transform transects
        transects_image_crs = {}
        for key_transect, transect in transects.items():
            transect_x_transformed, transect_y_transformed = transformer.transform(transect[:, 0], transect[:, 1])
            transects_image_crs[key_transect] = np.column_stack((transect_x_transformed, transect_y_transformed))

        # Make affine transformation matrix
        aff_mat = np.array([[dict_georef[key][1], dict_georef[key][2], dict_georef[key][0]],
                            [dict_georef[key][4], dict_georef[key][5], dict_georef[key][3]],
                            [0, 0, 1]])
        
        # Create affine transformation
        tform = transform.AffineTransform(aff_mat)

        # Transform polygon
        tmp = np.column_stack((x_poly, y_poly))
        points_converted = tform(tmp)
        poly_image_crs = shapely.geometry.Polygon(points_converted)

        # Calculate position on transect
        position_on_transect = {}
        gpd.GeoSeries(poly_image_crs.exterior).plot(ax=ax)

        for key_transect, transect in transects_image_crs.items():
            # Calculate intersection
            intersection = poly_image_crs.exterior.intersection(shapely.geometry.LineString(transect))

            # Calculate distance from bottom of transect
            point_transect = shapely.geometry.Point(transects[0][1, 0], transects[0][1, 1])

            if type(intersection) == shapely.geometry.LineString:
                position_on_transect[key_transect] = np.nan

            elif type(intersection) == shapely.geometry.Point:
                position_on_transect[key_transect] = point_transect.distance(intersection)

            if type(intersection) == shapely.geometry.MultiPoint:
                dist = [point_transect.distance(i) for i in intersection.geoms]
                position_on_transect[key_transect] = dist[np.argmin(dist)]
            
        # Save positions on transects
        dict_time_series[key] = position_on_transect

    return dict_time_series

def user_best_polygon(d_polygons, i_rgb, dict_choice={}):

    dates = list(d_polygons.keys())

    if dict_choice != {}:
        dict_choice_keys = list(dict_choice.keys())

        # Unique dates
        dates_unique = [date for date in dates if date not in dict_choice_keys]
    
    else:
        dates_unique = dates

    class Index:
        ind = 0

        def next(self, event):
            self.ind += 1
            i = self.ind % len(dates_unique)
            update_images(ax1, ax2, dates_unique[i])

    callback = Index()

    def on_button_click(label):
        dict_choice[dates_unique[callback.ind]] = label
        print(f"User chose: {label}")

    def update_images(ax1, ax2, date):
        ax1.clear()
        ax2.clear()

        ax1.imshow(i_rgb[date])
        ax2.imshow(i_rgb[date])

        gpd.GeoSeries(d_polygons[date]['polygon_opt'].exterior).plot(ax=ax1, color='k')

        colors = ['r', 'g', 'b', 'y', 'm', 'c']
        labels = ['NDWI', 'NDVI', 'NIR', 'NIR Otsu', 'NDWI Otsu', 'NDVI Otsu']

        idx = 0
        for key in d_polygons[date].keys():
            if key != 'polygon_opt':
                if d_polygons[date][key] is not None:
                    gpd.GeoSeries(d_polygons[date][key].exterior).plot(ax=ax2, color=colors[idx], label=labels[idx])
                idx += 1

        ax1.set_title(f'Image {date}')
        ax2.legend()
        plt.draw()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)

    update_images(ax1, ax2, dates_unique[0])

    button_ax = plt.axes([0.1, 0.1, 0.1, 0.075])
    button_ax2 = plt.axes([0.2, 0.1, 0.1, 0.075])
    button_ax3 = plt.axes([0.3, 0.1, 0.1, 0.075])
    button_ax4 = plt.axes([0.4, 0.1, 0.1, 0.075])
    button_ax5 = plt.axes([0.5, 0.1, 0.1, 0.075])
    button_ax6 = plt.axes([0.6, 0.1, 0.1, 0.075])
    button_ax7 = plt.axes([0.7, 0.1, 0.1, 0.075])
    button_ax8 = plt.axes([0.8, 0.1, 0.1, 0.075])

    button = Button(ax=button_ax, label='NDWI', color='r')
    button2 = Button(button_ax2, 'NDVI', color='g')
    button3 = Button(button_ax3, 'NIR', color='b')
    button4 = Button(button_ax4, 'NIR Otsu', color='y')
    button5 = Button(button_ax5, 'NDWI Otsu', color='m')
    button6 = Button(button_ax6, 'NDVI Otsu', color='c')
    button7 = Button(button_ax7, 'Optimal', color='grey')
    button8 = Button(button_ax8, 'Skip', color='white')

    button.on_clicked(lambda event: on_button_click('NDWI'))
    button2.on_clicked(lambda event: on_button_click('NDVI'))
    button3.on_clicked(lambda event: on_button_click('NIR'))
    button4.on_clicked(lambda event: on_button_click('NIR Otsu'))
    button5.on_clicked(lambda event: on_button_click('NDWI Otsu'))
    button6.on_clicked(lambda event: on_button_click('NDVI Otsu'))
    button7.on_clicked(lambda event: on_button_click('Optimal'))

    button.on_clicked(callback.next)
    button2.on_clicked(callback.next)
    button3.on_clicked(callback.next)
    button4.on_clicked(callback.next)
    button5.on_clicked(callback.next)
    button6.on_clicked(callback.next)
    button7.on_clicked(callback.next)
    button8.on_clicked(callback.next)

    # Display the interactive plot
    plt.show()

    button_ax._button = button
    button_ax2._button = button2
    button_ax3._button = button3
    button_ax4._button = button4
    button_ax5._button = button5
    button_ax6._button = button6
    button_ax7._button = button7
    button_ax8._button = button8

    return dict_choice


