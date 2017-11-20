import numpy as np
import random
from PIL import Image
from os import listdir
from os.path import isfile, join

"""
This is your object classifier. You should implement the train and
classify methods for this assignment.
"""
class ObjectClassifier():
    labels = ['Tree', 'Sydney', 'Steve', 'Cube']
    learned_model={}

    def testFeature(image_data):
        return True

    """
    checks: pixels with brightness >= 100 and orientation == 90
    against: pixels with brighness >= 100
    for: >20%
    """
    def feature_sad_panda(image_data):
        brightness = image_data[0]
        orientation = image_data[1]
        rows, columns = brightness.shape
        # Python: If you can't do it in one line, don't do it at all.
        # This is basically a DIY map-reduce, but using actual map-reduce would have been a bigger pain.
        edge_count=0
        response_count=0
        for y in range(rows):
            for x in range(columns):
                if brightness[y][x] >= 100:
                    edge_count += 1
                    if orientation[y][x] == 90:
                        response_count += 1
        print("Sad Panda:")
        print("\tresponse: " + str(response_count))
        print("\ttotal: " + str(edge_count))
        print("\t%: " + str(100.0 * response_count / edge_count))
        return 1.0 * response_count / edge_count >= .13


    """
    checks: pixels with brightness >=100
    against: all
    for: >10%
    """
    def feature_blue_squirrel(image_data):
        brightness = image_data[0]
        orientation = image_data[1]
        rows, columns = brightness.shape
        response = len([brightness[y][x] for y in range(rows) for x in range(columns) if brightness[y][x] >= 100])
        print("Blue Squirell:")
        print("\tresponse: " + str(response))
        print("\ttotal: " + str(rows*columns))
        print("\t%: " + str(100.0 * response / (rows * columns)))
        return 1.0 * response / (rows * columns) > .04



    features = [feature_sad_panda, feature_blue_squirrel]   # Will be names of predicate functions
    """
    Everytime a snapshot is taken, this method is called and
    the result is displayed on top of the four-image panel.
    """
    def classify(self, edge_pixels, orientations):
        return random.choice(self.labels)

    """
    This is your training method. Feel free to change the
    definition to take a directory name or whatever else you
    like. The load_image (below) function may be helpful in
    reading in each image from your datasets.
    """
    def train(self):
        matrix_prob = {label : {feature : 0 for feature in self.features} for label in self.labels}    # dict of dicts representing P(f|C)
        matrix_count = {label : {feature : 0 for feature in self.features} for label in self.labels}   # dict of dicts representing # times feature seen per class
        classes_seen = {label : 0 for label in self.labels}   # dict representing times each class encountered
        for classification in self.labels:
            data_folder = "./snapshots/training_data/" + classification + "/"
            images = [(data_folder + f) for f in listdir(data_folder) if isfile(join(data_folder, f))]
            for image in images:
                image_data = load_image(image)  # <-- this takes ~ .75 seconds. Slows everything down
                for feature in self.features:
                    # TODO check edge info against each feature
                    if feature(image_data):
                        matrix_count[classification][feature] += 1
                classes_seen[classification] += 1

            for classification in (f for f in self.labels if classes_seen[f] > 0):
                for feature in matrix_count[classification].keys():
                    matrix_prob[classification][feature] = 1.0 * matrix_count[classification][feature] / classes_seen[classification]
            self.learned_model = matrix_prob

    

"""
Loads an image from file and calculates the edge pixel orientations.
Returns a tuple of (edge pixels, pixel orientations).
"""
def load_image(filename):
    im = Image.open(filename)
    np_edges = np.array(im)
    upper_left = push(np_edges, 1, 1)
    upper_center = push(np_edges, 1, 0)
    upper_right = push(np_edges, 1, -1)
    mid_left = push(np_edges, 0, 1)
    mid_right = push(np_edges, 0, -1)
    lower_left = push(np_edges, -1, 1)
    lower_center = push(np_edges, -1, 0)
    lower_right = push(np_edges, -1, -1)
    vfunc = np.vectorize(find_orientation)
    orientations = vfunc(upper_left, upper_center, upper_right, mid_left, mid_right, lower_left, lower_center, lower_right)
    return (np_edges, orientations)


"""
Shifts the rows and columns of an array, putting zeros in any empty spaces
and truncating any values that overflow
"""
def push(np_array, rows, columns):
    result = np.zeros((np_array.shape[0],np_array.shape[1]))
    if rows > 0:
        if columns > 0:
            result[rows:,columns:] = np_array[:-rows,:-columns]
        elif columns < 0:
            result[rows:,:columns] = np_array[:-rows,-columns:]
        else:
            result[rows:,:] = np_array[:-rows,:]
    elif rows < 0:
        if columns > 0:
            result[:rows,columns:] = np_array[-rows:,:-columns]
        elif columns < 0:
            result[:rows,:columns] = np_array[-rows:,-columns:]
        else:
            result[:rows,:] = np_array[-rows:,:]
    else:
        if columns > 0:
            result[:,columns:] = np_array[:,:-columns]
        elif columns < 0:
            result[:,:columns] = np_array[:,-columns:]
        else:
            result[:,:] = np_array[:,:]
    return result

# The orientations that an edge pixel may have.
np_orientation = np.array([0,315,45,270,90,225,180,135])

"""
Finds the (approximate) orientation of an edge pixel.
"""
def find_orientation(upper_left, upper_center, upper_right, mid_left, mid_right, lower_left, lower_center, lower_right):
    a = np.array([upper_center, upper_left, upper_right, mid_left, mid_right, lower_left, lower_center, lower_right])
    return np_orientation[a.argmax()]
