# ADD DOCSTRINGS

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

def combine_geometry_extent(contour, img, distance_threshold=30):

    # Expand image array with nan values for extent contouring
    expanded_array = np.full((img.shape[0]+2, img.shape[1]+2), np.nan)
    expanded_array[1:img.shape[0]+1, 1:img.shape[1]+1] = img

    # Create a mask with nan values
    nan_mask = np.isnan(expanded_array)

    # Find contour of extent (boundary between nan and non-nan values)
    extent = measure.find_contours(nan_mask, 0.5)

    # Create shapely objects
    polygon_extent = shapely.geometry.Polygon(extent[0]-1) # remove 1 to account for the expansion
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
                segment2 = shapely.ops.linemerge(linestring_extent.difference(segment1))

            else:
                segment1 = shapely.geometry.LineString(linestring_extent.coords[index1:(index2+1)])
                segment2 = shapely.ops.linemerge(linestring_extent.difference(segment1))

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

def histogram_pixel_values(img_no_nan, ax, plot_results, method='smooting', sigma=10.):

    # Create linspace of pixel values between min and max
    X = np.linspace(min(img_no_nan), max(img_no_nan), 1000)

    # Create histogram distribution for smoothing
    if method == 'kde':
        kernel = scipy.stats.gaussian_kde(img_no_nan.flatten())
        histogram_distribution = kernel(X)
    
    elif method == 'smooting':
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
    
def segmentation(img, rgb, label, segmented_image, cmap, classes_to_consider=None, plot_results=False, animation=False, verbose=False):
    
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
    X_minima, X_peaks, X, histogram_distribution = histogram_pixel_values(img_no_nan, ax=ax3, plot_results=plot_results, method='smooting')

    if plot_results:
        associate_peaks_to_classification(img, segmented_image, label, X, histogram_distribution, X_peaks, cmap)

    # Create empty list of polygons
    polygons = []
    polygons_dict = {}

    # List of thresholds
    thresholds = X_minima # np.linspace(min(X), max(X), 20)

    # Loop over thresholds (minima of histogram)
    for idx_minimum, minimum in enumerate(list(reversed(thresholds))):

        # Find contours with skimage.measure.find_contours (Marching Squares algorithm)
        contours = measure.find_contours(img, minimum, fully_connected='high', positive_orientation='high')

        # Get Otsu contour if label is NIR, NDWI and NDVI
        if label in ['NIR', 'NDWI', 'NDVI', 'MNDWI']:
            contours_otsu = measure.find_contours(img, f_otsu_two_classes, fully_connected='high', positive_orientation='high')
            contours_combined_otsu = combine_contours(contours_otsu, img, verbose=verbose)
            #polygons_dict['otsu'] = {'label': label,
                                     #'threshold': f_otsu_two_classes,
                                     #'polygons': [shapely.geometry.Polygon(np.flip(c)) for c in contours_combined_otsu]}

        # Combine contours
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