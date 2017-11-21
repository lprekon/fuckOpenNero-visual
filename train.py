from classifier import *
from os import listdir
import pickle
import sys

def find_average_feature():

	buddy = ObjectClassifier()
	values = {}
	features={}
	for f in buddy.features.keys():
		values[f.__name__] = []
	for classification in buddy.labels:
		data_folder = "./snapshots/training_data/" + classification + "/"
		images = [(data_folder + f) for f in listdir(data_folder) if isfile(join(data_folder, f))]
		for image in images:
			image_data = load_image(image)  # <-- this takes ~ .75 seconds. Slows everything down
			for feature_func in buddy.features.keys():
				values[feature_func.__name__].append(feature_func(image_data))
	for feature in buddy.features.keys():
		print(feature.__name__)
		data = values[feature.__name__]
		median = data[len(data) // 2]
		print("\tmeadian: " + str(median))
		features[feature.__name__] = median
	print("\ndumping features to ./feature/f.pkl\n")
	with open('features/f.pkl', 'wb') as f:
            pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)


def main():
	buddy = ObjectClassifier()
	# print(buddy.serialize())
	buddy.train()
	print(buddy.learned_model)
	print(buddy.serialize())

	models = listdir("models")
	buddy.save_model("model_" + str(len(models)))

if __name__ == '__main__':
	if len(sys.argv) == 1:
		main()
	else:
		find_average_feature()
