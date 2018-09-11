
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
get_ipython().magic('matplotlib inline')


# In[2]:

train_img = np.load('../images/training1-images.npy')
mask_img = np.load('../images/training1-masks.npy')


# In[3]:

train_img.shape


# In[4]:

stack = 0

fig, axes = plt.subplots(1,2, figsize=(16,6))
axes[0].imshow(train_img[stack,:,:])
axes[1].imshow(mask_img[stack,:,:], cmap='jet');


# In[5]:

stack = 1

fig, axes = plt.subplots(1,2, figsize=(16,6))
axes[0].imshow(train_img[stack,:,:])
axes[1].imshow(mask_img[stack,:,:]);


# In[6]:

stack = 0

uniqueLabels, counts = np.unique(mask_img[stack,:,:], return_counts=True)


# In[7]:

plt.hist(counts[1:]);


# In[8]:

# exclude cells at boundary

fig, axes = plt.subplots(mask_img.shape[0], 1, figsize=(6,26));
for stack in range(mask_img.shape[0]):
    bdry1 = np.unique(mask_img[stack,0,:])
    bdry2 = np.unique(mask_img[stack,-1,:])
    bdry3 = np.unique(mask_img[stack,:,0])
    bdry4 = np.unique(mask_img[stack,:,-1])

    nonBdryLabels = np.setdiff1d(np.unique(mask_img[stack,:,:]),
                                 np.union1d(np.union1d(np.union1d(bdry1, bdry2), 
                                                       bdry3), 
                                            bdry4));

    nonBdryLabels

    uniqueLabels, counts = np.unique(mask_img[stack,:,:][np.isin(mask_img[stack,:,:],nonBdryLabels)], return_counts=True)
    axes[stack].hist(counts);


# In[10]:

def mask_stack_to_single_image(masks, checkpoint_id):
    """
    Merge a stack of masks containing multiple instances to one large image.
    Args:
        image: full fov np array
        masks: stack of masks of shape [h,w,n]. Note that image.shape != masks.shape, because the shape of the masks is the size of the inference call. Since we are doing inference in patches, the masks are going to be of size of the patch.
    Returns:
        image that is the same shape as the original raw image, containing all of the masks from the mask stack
    """
    image = np.zeros((masks.shape[0:2]),dtype=np.uint16) # image that contains all cells (0 bg, >0 is cell id)
    
    # switch shape to [num_masks, h, w] from [h, w, num_masks]
    masks = masks.astype(np.uint16) 
    #masks = np.moveaxis(masks, 0, -1)
    masks = np.moveaxis(masks, -1, 0) 
    
    #image_shape = masks.shape
    #image = np.zeros(image_shape[1:])
    #print(image.shape)
    num_masks = masks.shape[0] # shape = [n, h, w]
        
    for i in range(num_masks):
        current_mask = masks[i]
        image = add_mask_to_ids(image, current_mask, checkpoint_id)
        checkpoint_id += 1
        
    return image, checkpoint_id

def pad(arrays, reference, offsets):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros((reference[0],reference[1]), dtype=np.uint16)
    print('result:')
    print(result.shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + arrays.shape[dim]) for dim in range(arrays.ndim)]
    #print(insertHere)
    #print(arrays.shape)
    # Insert the array in the result at the specified offsets
    result[insertHere] = arrays
    return result


# In[12]:

print(image.shape)


# In[13]:

image = train_img[0,:,:]

cropsize = 150
padding = 5

stack = np.zeros((image.shape[0], image.shape[1],1),dtype=np.uint16) # make new image of zeros (exclude third dimension, not using rgb)
visited = np.zeros(image.shape[0:2])
num_times_visited = np.zeros(image.shape[0:2])

num_row = image.shape[0] # num rows in the image
num_col = image.shape[1]

assert cropsize < num_row and cropsize < num_col, 'cropsize must be smaller than the image dimensions'

#rowlist = np.concatenate(([0],np.arange(cropsize-padding, num_row, cropsize)))
#collist = np.concatenate(([0],np.arange(cropsize-padding, num_col, cropsize)))
checkpoint_id = 1
for row in np.arange(0, num_row, cropsize-padding): # row defines the rightbound side of box
    for col in np.arange(0, num_row, cropsize-padding): # col defines lowerbound of box
        masks_with_ids = np.zeros(image.shape[0:2])
        upperbound = row
        lowerbound = row + cropsize
        leftbound  = col
        rightbound = col + cropsize

        if lowerbound > num_row:
            lowerbound = num_row
            upperbound = num_row-cropsize

        if rightbound > num_col:
            rightbound = num_col
            leftbound  = num_col-cropsize
        #upperbound = bound(final_image, cropsize, padding, minsize, row, 'upper')
        #lowerbound = bound(final_image, cropsize, padding, minsize, row, 'lower')
        #rightbound = bound(final_image, cropsize, padding, minsize, col, 'right')
        #leftbound  = bound(final_image, cropsize, padding, minsize, col, 'left')
        #print(row)
        #print(col)
        print('bounds:')
        print('upper: {}'.format(upperbound))
        print('lower: {}'.format(lowerbound))
        print('left : {}'.format(leftbound))
        print('right: {}'.format(rightbound))

        num_times_visited[upperbound:lowerbound, leftbound:rightbound] += 1
        cropped_image = image[upperbound:lowerbound, leftbound:rightbound]
        #print('cropped image shape: {}'.format(cropped_image.shape))
        #print(cropped_image.shape)

        masks = mask_img[upperbound:lowerbound, leftbound:rightbound]

        #padded_masks = pad(masks, [num_row, num_col, masks.shape[2]], [upperbound, leftbound])
        #padded_masks = pad(masks, [num_row, num_col, masks.shape[2]], [upperbound,leftbound,0])
        #print('mask shape:')
        #print (padded_masks.shape)

        one_inference_mask_image, checkpoint_id = mask_stack_to_single_image(masks, checkpoint_id) # works
        padded_inference_mask = pad(one_inference_mask_image, [num_row, num_col], [upperbound,leftbound])
        padded_inference_mask = np.expand_dims(padded_inference_mask, axis=2)
        stack = np.concatenate((stack, padded_inference_mask), axis=2)
        


# In[93]:

#from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import csgraph_from_dense, connected_components


# In[98]:



stack = np.load('../images/inference-stack.npy')
stack = stack[:,:,1:]

# loop over non-zero labels along right (bottom) boundaries:
#   if label immediately to the left (below) is non-zero:
#     add connection between right-left (bottom-below) labels in the sparse connectivity matrix
# 
# keep the labels that don't touch the right (bottom) boundaries as they are; these label IDs are already taken and therefore "unavailable"
#
# find connected components in the graph
# sort the connected components based on the smallest ID within component
# for each connected component:
#   assign the next availabble label ID (the smallest of the labels being combined) to the combined component
#



# In[99]:

stack.shape


# In[100]:

fig, axes = plt.subplots(2,2,figsize=(12,12))
axes[0,0].imshow(stack[:,:,0])
axes[0,1].imshow(stack[:,:,1])
axes[1,0].imshow(stack[:,:,2])
axes[1,1].imshow(stack[:,:,3])


# In[101]:

np.unique(stack[:,:,0])


# In[102]:

np.unique(stack[:,:,1])


# In[103]:

nonzero_idx = np.any(stack,axis=2)

plt.imshow(nonzero_idx)


# In[104]:

nonzero_idx.sum()


# In[107]:

# get unique label combinations across stacks
labels_to_combine = np.unique(stack[nonzero_idx],axis=0)
labels_to_combine


# In[108]:

conn_mat = np.zeros((labels_to_combine.max()+1,labels_to_combine.max()+1), dtype='bool')


# In[109]:

conn_mat.shape


# In[112]:

#labels_to_combine = labels_to_combine[np.count_nonzero(labels_to_combine,axis=1)>1,:]

#label_idx1 =
#label_idx2 = 

for row, label_combo in enumerate(labels_to_combine):
    group = label_combo[np.nonzero(label_combo)]
    for i in range(len(group)-1):
        for j in range(i+1,len(group)):
            conn_mat[group[i], group[j]] = True
            conn_mat[group[j], group[i]] = True

np.fill_diagonal(conn_mat, True)


# In[113]:

plt.imshow(conn_mat)


# In[114]:

graph = csgraph_from_dense(conn_mat)


# In[115]:

n_conncomp, graph_complabels = connected_components(graph, directed=False)


# In[116]:

graph_complabels


# In[117]:

graph_complabels.shape


# In[120]:

len(np.unique(graph_complabels))


# In[118]:

len(np.unique(stack))


# In[121]:

np.any(stack==1, axis=2).shape


# In[122]:

result = np.zeros_like(stack[:,:,0])

for label in np.unique(stack):
    result[np.any(stack==label,axis=2)] = graph_complabels[label]


# In[126]:

plt.imshow(result, cmap='jet', )


# In[127]:

from PIL import Image


# In[133]:

output = Image.open('../inference-small-nuc/output.tif')
input_img = Image.open('../inference-small-nuc/input.tif')


# In[131]:

plt.imshow(np.array(output), cmap='jet')


# In[137]:

fig, axes= plt.subplots(1,2,figsize=(16,12))
axes[0].imshow(np.array(input_img))
axes[1].imshow(result, cmap='jet')


# In[142]:

plt.imshow(np.array(input_img), alpha=.5)
#plt.imshow(result, cmap='jet', alpha=.5)
plt.contour(result, 0.5, alpha=.5)


# In[143]:

plt.imshow(np.array(input_img), alpha=.5)
#plt.imshow(result, cmap='jet', alpha=.5)
plt.contour(np.array(output), 0.5, alpha=.5)


# In[ ]:



