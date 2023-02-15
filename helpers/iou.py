# define the Intersection over Union algorithm
# we use this ratio to measure the object detection accuracy, 
# including how much a given Selective Search proposal
# overlaps with a ground-truth bounding box (useful when we
# have to generate positive and negative examples from training)

def compute_iou(boxA, boxB):
    # determine the (x, y)-coordenates of the interseaction rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compyte the area of intersection rectangle
    # "+1" to count edges from A and B
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)  
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area.
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou