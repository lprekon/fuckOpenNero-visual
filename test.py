from classifier import *

def test():
	buddy = ObjectClassifier()
	values = {}

	for f in buddy.features.keys():
		values[f] = []
	right = {}
	right['total'] = 0
	wrong = {}
	wrong['total'] = 0
	for classification in buddy.labels:
		print ("\nTesting " + classification)
		right[classification] = 0
		wrong[classification] = 0
		data_folder = "./snapshots/validation_data/" + classification + "/"
		images = [(data_folder + f) for f in listdir(data_folder) if isfile(join(data_folder, f))]
		for image in images:
			image_data = load_image(image)  # <-- this takes ~ .75 seconds. Slows everything down
			result = buddy.classify(*image_data)
			if result.lower() == classification.lower():
				right['total'] += 1
				right[classification] += 1
				print("CORRECT! " + result)
			else:
				wrong['total'] += 1
				wrong[classification] += 1
				print("WRONG! " + result)
	print "\nRESULTS:"
	for c in sorted(right.keys())[::-1]:
		tot = right[c] + wrong[c]
		fmt = "%6s: %2d / %d = %3.3f%%" % (c, right[c], tot, 100.0 * float(right[c])/float(tot))
		print fmt
def main():
	test()

if __name__ == '__main__':
	main()
