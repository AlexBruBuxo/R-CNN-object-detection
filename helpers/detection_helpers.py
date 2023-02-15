
import imutils


"""
Image Pyramids:

Construct a multi-scale representation of an image to find objects in 
images at different scales.
"""
def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    # yield the original image
    yield image

    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum size,
        # then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


"""
Sliding Windows: 

Fixed-size rectangle that slides from left-to-right and top-to-bottom 
within an image. This is performed for each scale of the pyramid. At each 
stop of the window we would:
1. Extract the ROI (region of interest - i.e., window).
2. Pass it through our image classifier.
3. Obtain the output predictions.
Combined with image pyramids, sliding windows allow us to localize objects 
at different locations and multiple scales of the input image.

image = may come from the output of the image pyramid
step = step size (e.g., 4 to 8 pixels)
ws = window size (width and height)
"""
def sliding_window(image, step, ws):
    # slide a window acrross the image
    for y in range(0, image.shape[0] - ws[1], step): # rows
        for x in range(0, image.shape[1] - ws[0], step): # columns
            # yield the current window
            yield (x, y, image[y:y + ws[1], x:x + ws[0]]) # generator