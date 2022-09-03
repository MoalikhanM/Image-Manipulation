import numpy as np
import math
import random


def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START
    """
    largest_set = []
    i,j = random.choice(matched_pairs)
    #repeat the random selection 10 times and then select the largest consistent subset that was found
    
    for n in range(10): 
        #calculate the orientation change of the one match
        orientation_change = (keypoints1[i][1] - keypoints2[j][1]) * 180 / np.pi
        #calculate the range for other pairs orientation changes
        orientation_range_in = orientation_change - orient_agreement
        orientation_range_end = orientation_change + orient_agreement
        
        #calculate the scale ratio
        scale_ratio = keypoints2[i][2] / keypoints1[j][2]
        #calculate the scale ration range
        scale_range_in = scale_ratio - scale_agreement
        scale_range_end = scale_ratio + scale_agreement
        
        #find best matches
        for i, j in matched_pairs:
            #calculate the orientation change and scale ratio
            if (orientation_range_in < orientation_change and orientation_range_end > orientation_change
                and scale_range_in < scale_ratio and scale_range_end > scale_ratio):
                largest_set.append([i,j])
    """
    ranint = random.randint(0,len(matched_pairs))
    largest_set = []
    ranint = random.randint(0,len(matched_pairs)-1)
    x1,y2 = matched_pairs[ranint]
    radian1 = keypoints1[x1][3]+keypoints2[y2][3]
    for i in range(0,10):
        ranint = random.randint(0,len(matched_pairs)-1)
        x,y = matched_pairs[ranint]
        radian = keypoints1[x][3]+keypoints2[y][3]
        diff = radian1 - radian
        if math.cos(radian-radian1)>math.cos(orient_agreement):
            matched_pairs.remove(matched_pairs[ranint])
    for i in range(0,len(matched_pairs)):
        x,y = matched_pairs[i]
        if keypoints1[x][2]/keypoints2[y][2] >=0.5:
            if keypoints1[x][2]/keypoints2[y][2] <=1.5:
                largest_set.append([x,y])
    
    ## END
    assert isinstance(largest_set, list)
    return largest_set 


# if avoid "RANSACFilter", then this function is working
def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    ## START
    num = 5
    matched_pairs = []
    shape1 = descriptors1.shape[0]
    shape2 = descriptors2.shape[0]
    match = {}
    matched_pairs=[]
    for i in range(shape1):
        for j in range(shape2):
            angle = math.acos(np.dot(descriptors1[i],descriptors2[j]))
            match[angle] = i,j
        match_list = sorted(match)
        f_neigh = match_list[0]
        s_neigh = match_list[1]
        ratio = f_neigh/s_neigh
        if ratio <= threshold:
            key1,key2 = match[f_neigh]
            matched_pairs.append([key1,key2])
        match = {}
    """
    for i in range(len(descriptors1)): # firsly through descriptor of image1 
        best_angle = [] # angle array
        for d2 in descriptors2: # then compare it with descriptor of image 2
            # find that angle and calculate by the inverse cosine
            x = np.dot(descriptors1[i], d2)
            best_angle.append(math.acos(x)) 
        #sort best angles
        match_angle = sorted(best_angle) 

        #append if the ratio of two best angles is below a threshold.
        if (match_angle[0] / match_angle[1] <= threshold): 
            matched_pairs.append([best_angle.index(match_angle[0])])
    """
    ## END
    return matched_pairs


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac
