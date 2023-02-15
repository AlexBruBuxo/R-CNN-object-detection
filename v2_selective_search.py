"""
This is a test script to understand the use of Selective Search with OpenCV. 

Source: https://pyimagesearch.com/2020/06/29/opencv-selective-search-for-object-detection/  

This file can be executed as follows:
$ python v2_selective_search.py --image images/bear.jpg (--method quality)
"""

import argparse
import random
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
            help="path to the inout image")
ap.add_argument("-m", "--method", type=str, default="fast",
            help="selective search method") # fast / quality
args = vars(ap.parse_args())

# load the input image
image = cv2.imread(args["image"])

# initialize OpenCV's selective search implementation and set the
# input image
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)

# check to see if we are using the *fast* but *less accurate* version
# of selective search
if args["method"] == "fast":
	print("[INFO] using *fast* selective search")
	ss.switchToSelectiveSearchFast()
# otherwise we are using the *slower* but *more accurate* version
else:
	print("[INFO] using *quality* selective search")
	ss.switchToSelectiveSearchQuality()

start = time.time()
rects = ss.process()  # run Selective Search
end = time.time()

# show how along selective search took to run along with the total
# number of returned region proposals
print("[INFO] selective search took {:.4f} seconds".format(end - start))
print("[INFO] {} total region proposals".format(len(rects)))

# loop over the region proposals in chunks (so we can better
# visualize them)
for i in range(0, len(rects), 100):
	# clone the original image so we can draw on it
	output = image.copy()

	# loop over the current subset of region proposals
	for (x, y, w, h) in rects[i:i + 100]:
		# draw the region proposal bounding box on the image
		color = [random.randint(0, 255) for j in range(0, 3)]
		cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
    
	# show the output image
	cv2.imwrite("./images/results/v2_output.jpg", output)
	key = cv2.waitKey(0) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
