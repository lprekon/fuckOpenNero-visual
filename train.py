from classifier import *
from os import listdir

def main():
	buddy = ObjectClassifier()
	buddy.train()
	print(buddy.learned_model)
	print(buddy.serialize())
	models = listdir("models")
	f = open('./models/model_' + str(len(models)) + '.dat', 'w')
	f.write(buddy.serialize())
	f.close()

if __name__ == '__main__':
	main()

def find_average_feature():
	buddy = ObjectClassifier()
	values = {}
	for f in buddy.features.keys():
		values[f] = []
	for classification in buddy.labels:
		data_folder = "./snapshots/training_data/" + classification + "/"
		images = [(data_folder + f) for f in listdir(data_folder) if isfile(join(data_folder, f))]
		for image in images:
			image_data = load_image(image)  # <-- this takes ~ .75 seconds. Slows everything down
			for feature_func in buddy.features.keys():
				values[feature_func].append(feature_func(image_data))
	for feature in buddy.features.keys():
		print(feature)
		data = values[feature]
		mean = sum(data) / len(data)
		median = data[len(data) // 2]
		print("\tmean: " + str(mean))
		print("\tmeadian: " + str(median))
