#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pickle
import copy
import numpy as np
from dtw_seg import DTWSegmentation


# In[66]:


file_path = "../../data/spatail_mask.pkl"
spatail_masks=None
# Open the file in read-binary mode
with open(file_path, 'rb') as file:
    spatail_masks = pickle.load(file)


# In[67]:


file_path = "../../data/temporal_mask.pkl"
temporal_masks=None
# Open the file in read-binary mode
with open(file_path, 'rb') as file:
    temporal_masks = pickle.load(file)


# In[73]:


# generate the Spatail Circle Gap
circle_gap__spatail=spatail_masks["circle"]
separated_gap_spatail=spatail_masks["separated"]


# In[81]:


tempral_seprated_masks=temporal_masks['separated']
tempral_sequential_masks=temporal_masks['sequential']


# In[86]:


# generate the Spatail Circle Gap
directory_SENTINEL=r"../../../Images_Zone/Images_Zone/"
mask_30=tempral_seprated_masks[3]
bands=["B2","B3","B4","B8","B11","B12"]
cas_types=["spatail","temporal"]
gap_size_temp=["10%","20%","30%","40%","50%"]
spatail_gaps_types=["circle","separated"]
temporal_types=["sequential","separated"]
for band_2 in bands:
    for cas in cas_types:
        stack_images_bands[band_2][cas]={}
        for i in range(len(tempral_seprated_masks)):
            
            # first cas we need to to evalute is the spatail cas
            if cas =="spatail":
                # set the temporal gape size at 20%
                temporal_gap_size=mask_30
                for spatail_gap_type in spatail_gaps_types:
                    stack_images_bands[band_2][cas][spatail_gap_type]={}
                    # in case we have circle spatial gap
                    if spatail_gap_type == "circle":
                        for j in range(len(gap_size_temp)):
                            gap_size_tmp=gap_size_temp[j]
                            gap_spatail_size_mask=circle_gap__spatail[j]
                            dates_list=list(stack_images_bands[band_2]["ground_truth"].keys())
                            stack_images_bands[band_2][cas][spatail_gap_type][gap_size_tmp]={}
                            #apply the temporal Gap based on the Mask
                            for k in range(len(temporal_gap_size)):
                                tmp_temporal_gap_mask=temporal_gap_size[k]
                                ground_truth=copy.deepcopy(stack_images_bands[band_2]["ground_truth"][dates_list[k]])
                                
                                if(tmp_temporal_gap_mask):
                                    ground_truth[gap_spatail_size_mask]=np.nan
                                    stack_images_bands[band_2][cas][spatail_gap_type][gap_size_tmp][dates_list[k]]=ground_truth
                                else:
                                    stack_images_bands[band_2][cas][spatail_gap_type][gap_size_tmp][dates_list[k]]=ground_truth

                    # in case we have separated spatial gap
                    elif spatail_gap_type == "separated":
                        for j in range(len(gap_size_temp)):
                            gap_size_tmp=gap_size_temp[j]
                            gap_spatail_size_mask=separated_gap_spatail[j]
                            dates_list=list(stack_images_bands[band_2]["ground_truth"].keys())
                            stack_images_bands[band_2][cas][spatail_gap_type][gap_size_tmp]={}
                            #apply the temporal Gap based on the Mask
                            for k in range(len(temporal_gap_size)):
                                tmp_temporal_gap_mask=temporal_gap_size[k]
                                ground_truth=copy.deepcopy(stack_images_bands[band_2]["ground_truth"][dates_list[k]])
                                
                                if(tmp_temporal_gap_mask):
                                    ground_truth[gap_spatail_size_mask]=np.nan
                                    stack_images_bands[band_2][cas][spatail_gap_type][gap_size_tmp][dates_list[k]]=ground_truth
                                else:
                                    stack_images_bands[band_2][cas][spatail_gap_type][gap_size_tmp][dates_list[k]]=ground_truth

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# first cas we need to to evalute is the spatail cas
            if cas =="temporal":
                # set the spatail gape size at 20%
                spatail_gap_size=separated_gap_spatail[1]
                for temporal_gap_type in temporal_types:
                    stack_images_bands[band_2][cas][temporal_gap_type]={}
                    # in case we have circle spatial gap
                    if temporal_gap_type == "sequential":
                        for j in range(len(gap_size_temp)):
                            gap_size_tmp=gap_size_temp[j]
                            gap_temporal_size_mask=tempral_sequential_masks[j]
                            dates_list=list(stack_images_bands[band_2]["ground_truth"].keys())
                            stack_images_bands[band_2][cas][temporal_gap_type][gap_size_tmp]={}
                            #apply the temporal Gap based on the Mask
                            for k in range(len(temporal_gap_size)):
                                tmp_temporal_gap_mask=gap_temporal_size_mask[k]
                                ground_truth=copy.deepcopy(stack_images_bands[band_2]["ground_truth"][dates_list[k]])
                                
                                if(tmp_temporal_gap_mask):
                                    ground_truth[spatail_gap_size]=np.nan
                                    stack_images_bands[band_2][cas][temporal_gap_type][gap_size_tmp][dates_list[k]]=ground_truth
                                else:
                                    stack_images_bands[band_2][cas][temporal_gap_type][gap_size_tmp][dates_list[k]]=ground_truth

                    # in case we have separated spatial gap
                    elif temporal_gap_type == "separated":
                        for j in range(len(gap_size_temp)):
                            gap_size_tmp=gap_size_temp[j]
                            gap_temporal_size_mask=tempral_seprated_masks[j]
                            dates_list=list(stack_images_bands[band_2]["ground_truth"].keys())
                            stack_images_bands[band_2][cas][temporal_gap_type][gap_size_tmp]={}
                            #apply the temporal Gap based on the Mask
                            for k in range(len(temporal_gap_size)):
                                tmp_temporal_gap_mask=gap_temporal_size_mask[k]
                                ground_truth=copy.deepcopy(stack_images_bands[band_2]["ground_truth"][dates_list[k]])
                                
                                if(tmp_temporal_gap_mask):
                                    ground_truth[spatail_gap_size]=np.nan
                                    stack_images_bands[band_2][cas][temporal_gap_type][gap_size_tmp][dates_list[k]]=ground_truth
                                else:
                                    stack_images_bands[band_2][cas][temporal_gap_type][gap_size_tmp][dates_list[k]]=ground_truth




# In[87]:


def genrate_dataset(images_dict,band,gap_type,gap_sinario,gap_size):
    nan_mask=[]
    data_list=[]
    for date in stack_images_bands[band][gap_type][gap_sinario][gap_size].keys():
        image=stack_images_bands[band][gap_type][gap_sinario][gap_size][date]
        nan_mask.append(np.isnan(image).any())
        # Set pixels with value 0 to NaN
        image = np.where(image == 0, np.nan, image)
        data_list.append(image)
    return np.array(data_list) , nan_mask


# In[89]:


seg_dtw=DTWSegmentation(test_ds[0].shape,100,1)


# In[90]:


regions_result={}
gap_size_list=["10%","20%","30%","40%","50%"]
bands=["B2","B3","B4","B8","B11","B12"]
gap_types=['spatail', 'temporal']
gap_sinarios_list=['circle', 'separated', 'sequential', 'separated']
for band in bands:
    regions_result[band]={}
    for gap_type in gap_types:
        regions_result[band][gap_type]={}
        for type_ in gap_sinarios_list:
                regions_result[band][gap_type][type_]={}
                for gap_index , gap_size  in enumerate(gap_size_list):
                    regions_result[band][gap_type][type_][gap_size]=[]
                    image_time_series,nan_mask=genrate_dataset(stack_images_bands,band,gap_type,type_,gap_size)
                    threeshoulds=seg_dtw.calculate_similarity_threshold_mean(test_ds)
                    image_seeds=seg_dtw.select_seeds_from_edges_time_series(test_ds,100,10)
                    for image in image_time_series:
                        regions_result[band][gap_type][type_][gap_size].append(seg_dtw.run_segmentation_process(test_ds,nan_mask,image_seeds,threeshoulds))


# In[95]:


file_path="./output/segmentation_result.pkl"
with open(file_path, 'wb') as file:
    pickle.dump(regions_result, file)

