import numpy as np
import random
import re
from PIL import Image
from os import listdir
from os.path import isfile, join
import pickle

"""
This is your object classifier. You should implement the train and
classify methods for this assignment.
"""
class ObjectClassifier():

    def __init__(self):
        self.labels = ['Tree', 'Sydney', 'Steve', 'Cube']
        self.features = {}

        self.features[self.feature_happy_panda] = .24
        self.features[self.feature_grumpy_panda] = .5
        self.features[self.feature_sleepy_panda] = .55
        self.features[self.feature_dopey_panda] = .44
        self.features[self.feature_sad_panda] = .13
        self.features[self.feature_bashful_panda] = .04
        self.features[self.feature_sad_panda] = .13
        self.features[self.feature_sneezy_panda] = .3
        self.features[self.feature_blue_squirrel] = .019
        self.features[self.feature_flying_squirrel] = 0.19

        self.learned_model = {}

        if len(listdir("models")) > 0:
            self.learned_model = self.load_model(sorted(listdir("models"))[-1][:-4])
        else:
            self.learned_model = {label:{self.feature_to_name(feature_func):0 for feature_func in self.features.keys()} for label in self.labels}

    # [0,315,45,270,90,225,180,135]

    def feature_to_name(self, feature):
        mapping = {}
        for feature in self.features:
            mapping[feature] = feature.__name__
        return mapping[feature]


    def name_to_feature(self, name):
        mapping = {}
        for feature in self.features:
            mapping[feature.__name__] = feature
        return mapping[name]


    """
    checks: pixels with brightness >= 100 and orientation == 0
    against: pixels with brighness >= 100
    """
    def feature_happy_panda(self, image_data):
        brightness = image_data[0]
        orientation = image_data[1]
        rows, columns = brightness.shape
        # Python: If you can't do it in one line, don't do it at all.
        # This is basically a DIY map-reduce, but using actual map-reduce would have been a bigger pain.
        z = np.zeros(brightness.shape)
        ones = np.ones(brightness.shape)
        b = np.where(brightness >= 100, ones, z)
        # print(b)
        o = np.where(orientation == 0, ones, z)
        # print(o)
        both = np.logical_and(b == 1,o == 1)
        # print(both)
        edge_count = np.sum(b)
        response_count = np.sum(both)
        # edge_count=0
        # response_count=0
        # for y in range(rows):
        #     for x in range(columns):
        #         if brightness[y][x] >= 100:
        #             edge_count += 1
        #             if orientation[y][x] == 0:
        #                 response_count += 1
        return 1.0 * response_count / edge_count



    """
    checks: pixels with brightness >= 100 and orientation == 315
    against: pixels with brighness >= 100
    for: >20%
    """
    def feature_grumpy_panda(self, image_data):
        brightness = image_data[0]
        orientation = image_data[1]
        rows, columns = brightness.shape
        # Python: If you can't do it in one line, don't do it at all.
        # This is basically a DIY map-reduce, but using actual map-reduce would have been a bigger pain.
        z = np.zeros(brightness.shape)
        ones = np.ones(brightness.shape)
        b = np.where(brightness >= 100, ones, z)
        # print(b)
        o = np.where(orientation == 315, ones, z)
        # print(o)
        both = np.logical_and(b == 1,o == 1)
        # print(both)
        edge_count = np.sum(b)
        response_count = np.sum(both)
        # edge_count=0
        # response_count=0
        # for y in range(rows):
        #     for x in range(columns):
        #         if brightness[y][x] >= 100:
        #             edge_count += 1
        #             if orientation[y][x] == 315:
        #                 response_count += 1
        return 1.0 * response_count / edge_count



    """
    checks: pixels with brightness >= 100 and orientation == 45
    against: pixels with brighness >= 100
    for: >20%
    """
    def feature_sleepy_panda(self, image_data):
        brightness = image_data[0]
        orientation = image_data[1]
        rows, columns = brightness.shape
        # Python: If you can't do it in one line, don't do it at all.
        # This is basically a DIY map-reduce, but using actual map-reduce would have been a bigger pain.
        edge_count=0
        response_count=0
        z = np.zeros(brightness.shape)
        ones = np.ones(brightness.shape)
        b = np.where(brightness >= 100, ones, z)
        # print(b)
        o = np.where(orientation == 45, ones, z)
        # print(o)
        both = np.logical_and(b == 1,o == 1)
        # print(both)
        edge_count = np.sum(b)
        response_count = np.sum(both)
        # for y in range(rows):
        #     for x in range(columns):
        #         if brightness[y][x] >= 100:
        #             edge_count += 1
        #             if orientation[y][x] == 45:
        #                 response_count += 1
        return 1.0 * response_count / edge_count



    """
    checks: pixels with brightness >= 100 and orientation == 270
    against: pixels with brighness >= 100
    for: >20%
    """
    def feature_dopey_panda(self, image_data):
        brightness = image_data[0]
        orientation = image_data[1]
        rows, columns = brightness.shape
        # Python: If you can't do it in one line, don't do it at all.
        # This is basically a DIY map-reduce, but using actual map-reduce would have been a bigger pain.
        edge_count=0
        response_count=0
        z = np.zeros(brightness.shape)
        ones = np.ones(brightness.shape)
        b = np.where(brightness >= 100, ones, z)
        # print(b)
        o = np.where(orientation == 270, ones, z)
        # print(o)
        both = np.logical_and(b == 1,o == 1)
        # print(both)
        edge_count = np.sum(b)
        response_count = np.sum(both)
        # for y in range(rows):
        #     for x in range(columns):
        #         if brightness[y][x] >= 100:
        #             edge_count += 1
        #             if orientation[y][x] == 270:
        #                 response_count += 1
        return 1.0 * response_count / edge_count




    """
    checks: pixels with brightness >= 100 and orientation == 90
    against: pixels with brighness >= 100
    """
    def feature_sad_panda(self, image_data):
        brightness = image_data[0]
        orientation = image_data[1]
        rows, columns = brightness.shape
        # Python: If you can't do it in one line, don't do it at all.
        # This is basically a DIY map-reduce, but using actual map-reduce would have been a bigger pain.
        edge_count=0
        response_count=0
        z = np.zeros(brightness.shape)
        ones = np.ones(brightness.shape)
        b = np.where(brightness >= 100, ones, z)
        # print(b)
        o = np.where(orientation == 90, ones, z)
        # print(o)
        both = np.logical_and(b == 1,o == 1)
        # print(both)
        edge_count = np.sum(b)
        response_count = np.sum(both)
        # for y in range(rows):
        #     for x in range(columns):
        #         if brightness[y][x] >= 100:
        #             edge_count += 1
        #             if orientation[y][x] == 90:
        #                 response_count += 1
        return 1.0 * response_count / edge_count



    """
    checks: pixels with brightness >= 100 and orientation == 225
    against: pixels with brighness >= 100
    for: >20%
    """
    def feature_bashful_panda(self, image_data):
        brightness = image_data[0]
        orientation = image_data[1]
        rows, columns = brightness.shape
        # Python: If you can't do it in one line, don't do it at all.
        # This is basically a DIY map-reduce, but using actual map-reduce would have been a bigger pain.

        edge_count=0
        response_count=0
        z = np.zeros(brightness.shape)
        ones = np.ones(brightness.shape)
        b = np.where(brightness >= 100, ones, z)
        # print(b)
        o = np.where(orientation == 225, ones, z)
        # print(o)
        both = np.logical_and(b == 1,o == 1)
        # print(both)
        edge_count = np.sum(b)
        response_count = np.sum(both)
        # for y in range(rows):
        #     for x in range(columns):
        #         if brightness[y][x] >= 100:
        #             edge_count += 1
        #             if orientation[y][x] == 225:
        #                 response_count += 1
        return 1.0 * response_count / edge_count



    """
    checks: pixels with brightness >= 100 and orientation == 180
    against: pixels with brighness >= 100
    for: >20%
    """
    def feature_doc_panda(self, image_data):
        brightness = image_data[0]
        orientation = image_data[1]
        rows, columns = brightness.shape
        # Python: If you can't do it in one line, don't do it at all.
        # This is basically a DIY map-reduce, but using actual map-reduce would have been a bigger pain.
        edge_count=0
        response_count=0
        z = np.zeros(brightness.shape)
        ones = np.ones(brightness.shape)
        b = np.where(brightness >= 100, ones, z)
        # print(b)
        o = np.where(orientation == 180, ones, z)
        # print(o)
        both = np.logical_and(b == 1,o == 1)
        # print(both)
        edge_count = np.sum(b)
        response_count = np.sum(both)
        # for y in range(rows):
        #     for x in range(columns):
        #         if brightness[y][x] >= 100:
        #             edge_count += 1
        #             if orientation[y][x] == 180:
        #                 response_count += 1
        return 1.0 * response_count / edge_count



    """
    checks: pixels with brightness >= 100 and orientation == 135
    against: pixels with brighness >= 100
    for: >20%
    """
    def feature_sneezy_panda(self, image_data):
        brightness = image_data[0]
        orientation = image_data[1]
        # Python: If you can't do it in one line, don't do it at all.
        # This is basically a DIY map-reduce, but using actual map-reduce would have been a bigger pain.
        edge_count=0
        response_count=0
        z = np.zeros(brightness.shape)
        ones = np.ones(brightness.shape)
        b = np.where(brightness >= 100, ones, z)
        # print(b)
        o = np.where(orientation == 135, ones, z)
        # print(o)
        both = np.logical_and(b == 1,o == 1)
        # print(both)
        edge_count = np.sum(b)
        response_count = np.sum(both)
        # for y in range(rows):
        #     for x in range(columns):
        #         if brightness[y][x] >= 100:
        #             edge_count += 1
        #             if orientation[y][x] == 135:
        #                 response_count += 1
        return 1.0 * response_count / edge_count



    """
    checks: pixels with brightness >=100
    against: all
    """
    def feature_blue_squirrel(self, image_data):
        brightness = image_data[0]
        orientation = image_data[1]
        rows, columns = brightness.shape

        z = np.zeros(brightness.shape)
        ones = np.ones(brightness.shape)
        b = np.where(brightness >= 100, ones, z)
        # print(b)
        response = np.sum(b)
        # response = len([brightness[y][x] for y in range(rows) for x in range(columns) if brightness[y][x] >= 100])
        return 1.0 * response / (rows * columns)

    """
    checks: pixels with orientation = 0
    against: bottom half
    """
    def feature_flying_squirrel(self, image_data):
        orientation = image_data[1]
        r, c = orientation.shape

        z = np.zeros((r//2, c))
        ones = np.ones((r - r//2, c))
        bot_half = np.concatenate((z, ones), axis=0)
        o = np.where(orientation == 0, bot_half, np.zeros(orientation.shape))
        return 1.0 * np.sum(o) / (r * c)



    """
    Everytime a snapshot is taken, this method is called and
    the result is displayed on top of the four-image panel.
    """
    def classify(self, edge_pixels, orientations):
        guess_weight = {}
        for label in self.labels:
            guess_weight[label] = 1

        for feature_func in self.features.keys():
            if(feature_func((edge_pixels, orientations)) > self.features[feature_func]):
                for label in self.labels:
                    guess_weight[label] *= self.learned_model[label][feature_func.__name__]
        # for label in self.labels:
        #     guess_weight[label] = 1
        #     for feature_func in self.features.keys():
        #         if(feature_func((edge_pixels, orientations)) > self.features[feature_func]):
        #             guess_weight[label] *= self.learned_model[label][feature_func.__name__]
        best_guess = []
        max_val = 0
        for label in self.labels:
            if guess_weight[label] > max_val:
                max_val = guess_weight[label]
                best_guess = [label]
            elif guess_weight[label] == max_val:
                best_guess.append(label)
        # print (best_guess)
        return random.choice(best_guess)


    def apply_feature(self, feature_func, matrix_count, classification, image_data):
        print("Feature: " + feature_func.__name__)
        res = feature_func(image_data)
        thread_lock.acquire(1)
        if  res > self.features[feature_func]:
            matrix_count[classification][feature_func.__name__] += 1
        thread_lock.release()


    """
    This is your training method. Feel free to change the
    definition to take a directory name or whatever else you
    like. The load_image (below) function may be helpful in
    reading in each image from your datasets.
    """
    def train(self):
        matrix_prob = {label : {feature.__name__ : 0 for feature in self.features} for label in self.labels}    # dict of dicts representing P(f|C)
        matrix_count = {label : {feature.__name__ : 0 for feature in self.features} for label in self.labels}   # dict of dicts representing # times feature seen per class
        classes_seen = {label : 0 for label in self.labels}   # dict representing times each class encountered
        print "Training"
        for classification in self.labels:
            print "--" + classification
            data_folder = "./snapshots/training_data/" + classification + "/"
            images = [(data_folder + f) for f in listdir(data_folder) if isfile(join(data_folder, f))]
            i = 0
            for image in images:
                print ("Starting: " + str(i))
                i += 1
                image_data = load_image(image)  # <-- this takes ~ .75 seconds. Slows everything down

                for feature_func in self.features.keys():
                    # TODO check edge info against each feature
                    if feature_func(image_data) > self.features[feature_func]:
                        matrix_count[classification][feature_func.__name__] += 1

                classes_seen[classification] += 1

            for classification in (f for f in self.labels if classes_seen[f] > 0):
                for feature in matrix_count[classification].keys():
                    matrix_prob[classification][feature] = 1.0 * matrix_count[classification][feature] / classes_seen[classification]
            self.learned_model = matrix_prob

    def serialize(self):
        label_column_size = reduce(max, map(len, self.labels), 0)
        feature_names = {feature_func: feature_func.__name__ for feature_func in self.features.keys()}
        feature_column_size = reduce(max, map(len, feature_names.values()), 0) + 1
        bits = ["".join((" " for i in range(feature_column_size + 1))),]
        for label in self.labels:
            bit = "".join(" " for i in range(label_column_size - len(label))) + label
            bits.append(bit)
        rows = [" ".join(bits),]
        for feature_func in self.features:
            row = feature_names[feature_func] + "".join(" " for i in range(feature_column_size - len(feature_names[feature_func]) + 1))
            for label in self.labels:
                data = str(self.learned_model[label][feature_names[feature_func]])
                row += "".join(" " for i in range(label_column_size - len(data) + 1))
                row += data
            rows.append(row)
        return "\n".join(rows)

    def save_model(self, name ):
        with open('models/'+ name + '.pkl', 'wb') as f:
            pickle.dump(self.learned_model, f, pickle.HIGHEST_PROTOCOL)

    def load_model(self, name):
        print("loading model " + name)
        with open('models/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)



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
