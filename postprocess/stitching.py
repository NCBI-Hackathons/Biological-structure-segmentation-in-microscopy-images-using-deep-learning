
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from PIL import Image


# In[2]:

def stitch(stack, numpix_threshold=0):
    from scipy.sparse.csgraph import csgraph_from_dense, connected_components
    nonzero_idx = np.any(stack,axis=2)
    
    # get unique label combinations across stacks
    labels_to_combine = np.unique(stack[nonzero_idx],axis=0)
    
    conn_mat = np.zeros((labels_to_combine.max()+1,labels_to_combine.max()+1), dtype='bool')
    
    for row, label_combo in enumerate(labels_to_combine):
        group = label_combo[np.nonzero(label_combo)]
        for i in range(len(group)-1):
            for j in range(i+1,len(group)):
                conn_mat[group[i], group[j]] = True
                conn_mat[group[j], group[i]] = True

    np.fill_diagonal(conn_mat, True)
    
    graph = csgraph_from_dense(conn_mat)
    
    n_conncomp, graph_complabels = connected_components(graph, directed=False)
    
    result = np.zeros_like(stack[:,:,0])
    
    for label in np.unique(stack):
        mask = np.any(stack==label,axis=2)
        if mask.sum() > numpix_threshold:
            result[np.any(stack==label,axis=2)] = graph_complabels[label]
    
    return result


def stitch_sparse(stack, numpix_threshold=0):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    
    # get unique label combinations across stacks
    nonzero_idx = np.any(stack,axis=2)
    labels_to_combine = stack[nonzero_idx]  
    
    combo_dict = {}
    for row, label_combo in enumerate(labels_to_combine):
        group = label_combo[np.nonzero(label_combo)] # this should already be sorted
        for i in range(len(group)):
            #if (group[i], group[i]) in combo_dict:
            #    combo_dict[(group[i], group[i])] += 1
            #else:
            #    combo_dict[(group[i], group[i])] = 1
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


# # inference-sparce-512

# In[3]:

stack = np.load('../inference-sparce-512/inference-bigger-cell.npy')
stack = stack[:,:,1:]
npix = 10


# In[4]:

result = stitch(stack, numpix_threshold=npix)


# In[6]:

result_sparse = stitch_sparse(stack, numpix_threshold=npix)


# In[7]:

Image.fromarray(result).save('../inference-sparce-512/result_hackathon.tif')
Image.fromarray(result_sparse).save('../inference-sparce-512/result_sparse_hackathon.tif')


# # inference-dense-512

# In[13]:

stack = np.load('../inference-dense-512/inference-stack.npy')
stack = stack[:,:,1:]

npix = 10


# In[14]:

result = stitch(stack, numpix_threshold=npix)


# In[15]:

result_sparse = stitch_sparse(stack, numpix_threshold=npix)


# In[16]:

Image.fromarray(result).save('../inference-dense-512/result_hackathon.tif')
Image.fromarray(result_sparse).save('../inference-dense-512/result_sparse_hackathon.tif')


# # inference-dense-512 using flipped

# In[17]:

stack = np.load('../inference-dense-512/inference-bigger-cell-after-flipping.npy')
stack = stack[:,:,1:]

npix = 10


# In[18]:

result = stitch(stack, numpix_threshold=npix)


# In[19]:

result_sparse = stitch_sparse(stack, numpix_threshold=npix)


# In[20]:

Image.fromarray(result).save('../inference-dense-512/result_flipped_hackathon.tif')
Image.fromarray(result_sparse).save('../inference-dense-512/result_flipped_sparse_hackathon.tif')


# # inference-dense-512 using both

# In[21]:

stack = np.load('../inference-dense-512/inference-stack.npy')
stack = stack[:,:,1:]

stack_flipped = np.load('../inference-dense-512/inference-bigger-cell-after-flipping.npy')
stack_flipped = stack_flipped[:,:,1:]

stack = np.concatenate((stack,stack_flipped), axis=2)

npix = 10


# In[ ]:

result = stitch(stack, numpix_threshold=npix)


# In[ ]:

result_sparse = stitch_sparse(stack, numpix_threshold=npix)


# In[ ]:

Image.fromarray(result).save('../inference-dense-512/result_both_hackathon.tif')
Image.fromarray(result_sparse).save('../inference-dense-512/result_both_sparse_hackathon.tif')


# In[ ]:



