# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage.io
import skimage.segmentation


path='../../nuclei_data'

data_stack = np.load(path+'/images/training1-images.npy')
truth = data_stack[0,:,:]

labels_stack = np.load(path+'/images/training1-masks.npy')
labels = labels_stack[0,:,:]
# labels, _, _ = skimage.segmentation.relabel_sequential(labels) # Relabel objects

# old way:
# y_pred = Image.open(path+'/inference-sparce-512/original_cell_inf.tif')

# new way
y_pred = Image.open(path+'/inference-sparce-512/result_hackathon.tif')

y_pred = np.array(y_pred)
y_pred, _, _ = skimage.segmentation.relabel_sequential(y_pred) # Relabel objects

# Show label image
fig = plt.figure()
plt.imshow(truth)
plt.title("Original image")

fig = plt.figure()
plt.imshow(labels)
plt.title("Ground truth labels")

# Show predictions
fig = plt.figure()
plt.imshow(y_pred)
plt.title("Prediction")

# Compute number of objects
true_objects = len(np.unique(labels))
pred_objects = len(np.unique(y_pred))
print("Number of true objects:", true_objects)
print("Number of predicted objects:", pred_objects)

# Compute intersection between all objects
intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

print np.shape(intersection)

# Compute areas (needed for finding the union between all objects)
area_true = np.histogram(labels, bins = true_objects)[0]
area_pred = np.histogram(y_pred, bins = pred_objects)[0]
area_true = np.expand_dims(area_true, -1)
area_pred = np.expand_dims(area_pred, 0)

# Compute union
union = area_true + area_pred - intersection

# Exclude background from the analysis
intersection = intersection[1:,1:]
union = union[1:,1:]
union[union == 0] = 1e-9

# Compute the intersection over union
iou = intersection / union

# Show iou
fig = plt.figure()
plt.imshow(iou)
plt.colorbar()
plt.title("Intersection over Union")


# Precision helper function
def precision_at(threshold, iou):
    matches = iou > threshold
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn

# Loop over IoU thresholds
prec = []
print("Thresh\tTP\tFP\tFN\tPrec.")
for t in np.arange(0.5, 1.0, 0.05):
    tp, fp, fn = precision_at(t, iou)
    p = tp / float(tp + fp + fn)
    print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
    prec.append(p)

print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))

plt.show()
