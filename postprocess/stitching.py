
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from PIL import Image


# In[2]:

def stitch(stack, numpix_threshold=0):
    '''
    Combine multiple instance segmentations based on overlapping patches into a single
    segmentation
    
    Args
    ----
        stack : np.ndarray
            first two dimensions of stack should be the dimensions of the input image,
            and the third dimension be the number of overlapping patches
        numpix_threshold : int
            a label will be retained in the output only if it has at least 
            numpix_threshold pixels
    
    Returns
    -------
        result : numpy.ndarray
            a 2-D array labels
    '''
    from scipy.sparse.csgraph import csgraph_from_dense, connected_components
    
    # find foreground labels
    nonzero_idx = np.any(stack,axis=2)
    
    # get unique label combinations across patches in stack
    labels_to_combine = np.unique(stack[nonzero_idx],axis=0)
    
    # compute a "connectivity matrix" that indicates which labels overlap across patches
    conn_mat = np.zeros((labels_to_combine.max()+1,labels_to_combine.max()+1), dtype='bool')
    
    for row, label_combo in enumerate(labels_to_combine):
        group = label_combo[np.nonzero(label_combo)]
        for i in range(len(group)-1):
            for j in range(i+1,len(group)):
                conn_mat[group[i], group[j]] = True
                conn_mat[group[j], group[i]] = True

    #np.fill_diagonal(conn_mat, True)
    
    # find connected components using this connectivity matrix
    # each connected component will be a different label in the result (as long as it
    # contains the minimum required number of pixels)
    graph = csgraph_from_dense(conn_mat)
    n_conncomp, graph_complabels = connected_components(graph, directed=False)
    
    result = np.zeros_like(stack[:,:,0])
    
    # reassign labels to the ids of the connected components
    for label in np.unique(stack):
        # get 2-D mask of voxels with a given label
        mask = np.any(stack==label,axis=2)
        
        # make sure that there are enough many pixels
        if mask.sum() > numpix_threshold:
            # if so, reassign this label to its corresponding connected component id
            result[np.any(stack==label,axis=2)] = graph_complabels[label]
    
    return result


def stitch_sparse(stack, numpix_threshold=0):
    '''
    Combine multiple instance segmentations based on overlapping patches into a single
    segmentation. This implementation uses a sparse instead of a dense connectivity matrix,
    so could be helpful if there is a large number of objects being segmented.
    
    Args
    ----
        stack : numpy.ndarray
            first two dimensions of stack should be the dimensions of the input image,
            and the third dimension be the number of overlapping patches
        numpix_threshold : int
            a label will be retained in the output only if it has at least 
            numpix_threshold pixels
    
    Returns
    -------
        result : numpy.ndarray
            a 2-D array labels
    '''
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    
    # get label combinations across stacks
    nonzero_idx = np.any(stack,axis=2)
    labels_to_combine = stack[nonzero_idx]  
    
    # keep track of the number of times a label combination occurs
    combo_dict = {}
    for row, label_combo in enumerate(labels_to_combine):
        group = label_combo[np.nonzero(label_combo)]
        for i in range(len(group)):
            for j in range(i+1,len(group)):
                if (group[i], group[j]) in combo_dict:
                    combo_dict[(group[i], group[j])] += 1
                else:
                    combo_dict[(group[i], group[j])] = 1
                       
    conn_mat = csr_matrix((np.ones(len(combo_dict), dtype='bool'),
                           ([key[0] for key in combo_dict.keys()], 
                            [key[1] for key in combo_dict.keys()])),
                          shape=(labels_to_combine.max()+1,labels_to_combine.max()+1))
    
    n_conncomp, graph_complabels = connected_components(conn_mat, directed=False)
    
    result = np.zeros_like(stack[:,:,0])

    for label in np.unique(stack):
        mask = np.any(stack==label,axis=2)
        if mask.sum() > numpix_threshold:
            result[np.any(stack==label,axis=2)] = graph_complabels[label]
    
    return result

def fix_euler_numbers(result, max_hole_size=999):
    '''
    Fix labels whose Euler numbers are not 1 (i.e., labels with holes or handles)
    
    Args
    ----
        result : numpy.ndarray
            2-D integer array of labeled objects
        max_hole_size : int
            max number of pixels to fill
    
    Returns
    -------
        result : numpy.ndarray
    '''
    from skimage.measure import regionprops
    from skimage.morphology import remove_small_holes
    
    # use skimage's regionprops to get the Euler number for each object
    props = regionprops(result)
    labels = np.array([roi['label'] for roi in props])
    euler_numbers = np.array([roi['euler_number'] for roi in props])
    
    for bad_label in labels[euler_numbers!=1]:
        mask = remove_small_holes(result==bad_label, area_threshold=max_hole_size)
        
        # if there are other labels that intersect with the holes that were filled,
        # then those other labels will be removed from the image
        result[np.isin(result,np.setdiff1d(np.unique(result[mask]),[0, bad_label]))] = 0
        result[mask] = bad_label
    
    return result

def split_large_objects(result, stack):
    '''
    Detect objects that are too large
    Determine whether these are actually single or multiple objects
    If multiple objects, manipulate labels accordingly
    
    Args
    ----
        result : numpy.ndarray
            2-D integer array of labeled objects
        stack : np.ndarray
            stack that was used to generate result
            first two dimensions of stack should be the dimensions of the input image,
            and the third dimension be the number of overlapping patches
    
    Returns
    -------
        result : numpy.ndarray
    '''
    pass


# # inference-sparce-512

# In[3]:

stack = np.load('../inference-sparce-512/inference-bigger-cell.npy')
stack = stack[:,:,1:]
npix = 10


# In[4]:

result = stitch(stack, numpix_threshold=npix)
result = fix_euler_numbers(result)


# In[5]:

result_sparse = stitch_sparse(stack, numpix_threshold=npix)
result_sparse = fix_euler_numbers(result_sparse)


# In[6]:

Image.fromarray(result).save('../inference-sparce-512/result_hackathon.tif')
Image.fromarray(result_sparse).save('../inference-sparce-512/result_sparse_hackathon.tif')


# # inference-dense-512

# In[7]:

stack = np.load('../inference-dense-512/inference-stack.npy')
stack = stack[:,:,1:]

npix = 10


# In[8]:

result = stitch(stack, numpix_threshold=npix)
result = fix_euler_numbers(result)


# In[9]:

Image.fromarray(result).save('../inference-dense-512/result_hackathon.tif')


# # inference-dense-512 using flipped

# In[10]:

stack = np.load('../inference-dense-512/inference-bigger-cell-after-flipping.npy')
stack = stack[:,:,1:]

npix = 10


# In[11]:

result = stitch(stack, numpix_threshold=npix)
result = fix_euler_numbers(result)


# In[12]:

Image.fromarray(result).save('../inference-dense-512/result_flipped_hackathon.tif')


# # inference-dense-512 using both

# In[13]:

stack = np.load('../inference-dense-512/inference-stack.npy')
stack = stack[:,:,1:]

stack_flipped = np.load('../inference-dense-512/inference-bigger-cell-after-flipping.npy')
stack_flipped = stack_flipped[:,:,1:]

stack_flipped[stack_flipped>0] += stack.max() # ensure that IDs are different 

stack = np.concatenate((stack,stack_flipped), axis=2)

npix = 10


# In[14]:

result = stitch(stack, numpix_threshold=npix)
result = fix_euler_numbers(result)


# In[15]:

Image.fromarray(result).save('../inference-dense-512/result_both_hackathon.tif')


# In[ ]:



