"""
    Core Utilities
    Supplements VIT-EKF by providing utilities for image extraction, processing, triangulation, etc.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage as ski # Template Matching
from types import SimpleNamespace

# ---------------------------------------------------------------------------------------------------------------------------------------
# Calibrated Camera Matrixes (Stereo Pair, 0.54m apart), from 2011_09_26 recording
P_Left = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00],
                   [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                   [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])

P_Right = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, -3.875744e+02],
                    [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                    [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])

kMaxCorners = 100

# params for Shi-Tomasi corner detection
feature_params = dict( maxCorners = kMaxCorners,
                       qualityLevel = 0.3,
                       minDistance = 20,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (45,45),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Measurement Covariance Matrix
R = 5*np.eye(2) # Measurement Covariance (2,2) for Mono Cam
R_stereo = 5*np.eye(4) # Measurement Covariance (4,4) for Stereo Cam

kStereoMatchWindowSize = 51
kHomogenousThreshold   = 0.001 # Triangulated Points must have Homogenous Coordinates greater than this (i.e. to avoid points too close to infinity)
kNumCorrectionCycles   = 20
kResidualThreshold     = 2 # Pixels
kCounter               = 0 # For changing ROI in image for new Tracks
# ---------------------------------------------------------------------------------------------------------------------------------------

def GetImagePair(data_path, index):
    """Get KITTI Left\Right GreyScale Image Pair
       
    Args:
        data_path: path to root directory of the recording.
        index: index of sample withing recording

    Returns:
        Left & Right Image
    """
    width = 10
    padded_number = str(index).zfill(width)
    print(padded_number)
    #data_path = '../' + data_path
    image_name_left   = data_path + 'image_00/data/' + padded_number + '.png'
    image_name_right  = data_path + 'image_01/data/' + padded_number + '.png'
    # 
    image_left = plt.imread(image_name_left)
    image_left = (image_left * 255).astype(np.uint8)
    image_right = plt.imread(image_name_right)
    image_right = (image_right * 255).astype(np.uint8)
    #
    images = SimpleNamespace()
    images.left = image_left
    images.right = image_right

    return images

def ComputePointCorrespondences(left_image, right_image, left_poi,
                                kernel_size=15, max_offset=100, return_scores=False):
    """Find, for each left-image point, its best match on the same row of the right image.

    Args:
        left_image, right_image: 2D numpy arrays (rows x cols), grayscale, same shape.
        left_poi: [2, N] array. Row 0 = x (column), row 1 = y (row) of each point.
        kernel_size: side length (pixels) of the square template. Forced to be odd.
        max_offset: the match is searched only in columns [x - max_offset, x] of
            the right image (the right camera is shifted right, so points move left).
        return_scores: if True, also return the N correlation scores (-1 .. 1).

    Returns:
        right_poi: [2, N] float array. right_poi[1] equals the (rounded and clipped)
            left y; right_poi[0] is the matched column, with 0 <= right_x <= left_x.
        scores (only if return_scores): [N] array of normalized cross-correlation
            values at the chosen match (near 1 = confident, near 0 = ambiguous).

    Edge handling: near an image border the template is clipped to the pixels that
    exist (no invented padding on the template), so a point on the top, bottom or
    right edge is matched with a smaller, off-centre window. The search range is
    clipped to the image, so a point at column 5 is searched only in columns 0..5
    (the right image is padded on its left side with its mean intensity so the
    clipped template can still be slid to those positions). Every input point gets
    a valid output point. Matches made from heavily clipped windows are less
    reliable; check the score.
    """
    left_image = np.asarray(left_image)
    right_image = np.asarray(right_image)
    if left_image.ndim != 2 or left_image.shape != right_image.shape:
        raise ValueError("left_image and right_image must be 2D arrays of the same shape")
    left_poi = np.asarray(left_poi, dtype=float)
    if left_poi.ndim == 1:
        left_poi = left_poi.reshape(2, 1)
    if left_poi.ndim != 2 or left_poi.shape[0] != 2:
        raise ValueError("left_poi must have shape [2, N]")

    n_rows, n_cols = left_image.shape
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1                      # centred window needs an odd size
    h = k // 2

    left_f = left_image.astype(np.float32)
    # Pad only the LEFT side of the right image, so a clipped template can be placed
    # with its left edge at a negative column when the point is near the left border.
    right_f = np.pad(right_image.astype(np.float32), ((0, 0), (h, 0)),
                     mode="constant", constant_values=float(right_image.mean()))

    xs = np.clip(np.rint(left_poi[0]), 0, n_cols - 1).astype(int)
    ys = np.clip(np.rint(left_poi[1]), 0, n_rows - 1).astype(int)

    right_poi = np.empty((2, xs.size), dtype=float)
    scores = np.empty(xs.size, dtype=float)

    for i, (x, y) in enumerate(zip(xs, ys)):
        # Template: k x k patch around (x, y), clipped to the image bounds.
        y0, y1 = max(0, y - h), min(n_rows, y + h + 1)
        x0, x1 = max(0, x - h), min(n_cols, x + h + 1)
        template = left_f[y0:y1, x0:x1]
        ox = x - x0                                   # point's column inside the template
        tw = x1 - x0

        c_min = max(0, x - int(max_offset))           # leftmost allowed match column
        # A match at column c puts the template's left edge at e = c - ox
        # (padded-image column e + h). Candidates c = c_min .. x.
        strip = right_f[y0:y1, (c_min - ox) + h:(x - ox) + h + tw]

        ncc = ski.feature.match_template(strip, template)[0]      # length x - c_min + 1; index j <-> column c_min + j
        ncc = np.nan_to_num(ncc, nan=-1.0)

        # argmax over the reversed array so ties (e.g. textureless patches) resolve to
        # the smallest offset (rightmost candidate) rather than the largest.
        j = ncc.size - 1 - int(np.argmax(ncc[::-1]))

        right_poi[0, i] = c_min + j
        right_poi[1, i] = y
        scores[i] = ncc[j]

    return (right_poi, scores) if return_scores else right_poi

def TriangulateFeaturesRemoveInvalidPoints(pts_2d_left, 
                                           pts_2d_right, 
                                           homog_coord_thresh = 0.001):
    """Triangulate 2D point pairs from Left\Right images into 3D space Remove points too close to infinity.
       
    Args:
        pts_2d_left: 
        pts_2d_right:
        homog_coord_thresh:

    Returns:
        Homogenous 3D Points [4xN] ready for plotting.
    """
    assert ((pts_2d_left.shape == pts_2d_right.shape) and 
            (pts_2d_left.shape[0] == 2))
    #
    points_4d_homo = cv2.triangulatePoints(P_Left, P_Right, pts_2d_left, pts_2d_right)
    # 3. Extract the 4th coordinate (w) for all points
    w_coordinates = points_4d_homo[3, :]
    # 4. Create a boolean mask where |w| >= 0.1
    valid_mask = np.abs(w_coordinates) >= homog_coord_thresh
    # 5. Filter out invalid points from the homogeneous array
    filtered_homo = points_4d_homo[:, valid_mask]
    pts_2d_left  = pts_2d_left[:,valid_mask]
    pts_2d_right = pts_2d_right[:,valid_mask]
    # 6. Convert remaining valid points to 3D Euclidean space (shape: 3 x M)
    points_3d = filtered_homo / filtered_homo[3, :]
    assert np.all(points_3d[-1,:] == 1.0) # Is Homogenized
    points_3d = points_3d[:-1,:] # Remove Homogenous Coordinate
    # Ensure no points are behind the Camera
    i_negative_points = points_3d[2,:] < 0
    assert (i_negative_points.sum() == 0)

    assert ((points_3d.shape[0] == 3) and 
            (points_3d.shape[1] == pts_2d_left.shape[1]) and 
            (pts_2d_left.shape == pts_2d_right.shape))

    return points_3d, pts_2d_left, pts_2d_right

def GetFeatureROIMask(image):
    """Binary Mask defines search region for Features.
       
    Args:
        image: defines shape of ROI-Mask

    Returns:
        Binary Mask
    """
    feature_roi_mask = np.zeros_like(image)
    feature_roi_mask[7:180,100:] = 1

    return feature_roi_mask

def GetFeatureROIMask_Dynamic(image, image_points):
    """Binary Mask defines search region for Features.
       
    Args:
        image: defines shape of ROI-Mask

    Returns:
        Binary Mask
    """
    global kCounter
    num_rows, num_cols = image.shape
    feature_roi_mask = np.zeros_like(image)
    num_sections = 4
    kStartColumn = 310 #100
    kEndColumn = 932 # num_cols-1
    values = np.linspace(kStartColumn, kEndColumn, num=(num_sections+1)).astype(int)
    i_sector = kCounter % num_sections
    feature_roi_mask[7:180,values[i_sector]:values[i_sector+1]] = 1
    # Zero-out the Mast at pre-existing tracks
    img_pt_cols = (image_points[0,:]>values[i_sector]) & (image_points[0,:]<values[i_sector+1])
    image_points = image_points[:,img_pt_cols].astype(int)
    radius = 10
    for i_point in range(image_points.shape[1]):
        col = image_points[0, i_point]
        row = image_points[1, i_point]
        # Calculate boundary limits safely clipped to image dimensions
        r_min = max(0, row - radius)
        r_max = min(num_rows, row + radius + 1)
        c_min = max(0, col - radius)
        c_max = min(num_cols, col + radius + 1)    
        # Zero out the bounding box area
        feature_roi_mask[r_min:r_max, c_min:c_max] = 0
    #
    kCounter += 1
    print('kCounter ', kCounter, ' end_col, ', values[i_sector+1])
    return feature_roi_mask

def ProjectPoints(points_3d, P_Cam):
    """Project Points from Homogenous-3D to Homogenous-2D
       
    Args:
        points_3d: [4xN] Homogenous 3D coordinates
        P_Cam: [3x4] Projection\Camera Matrix

    Returns:
        Homogenous 2D Points [3xN] ready for plotting.
    """
    assert (points_3d.shape[0] == 3) and (points_3d.ndim == 2) # and np.all(points_3d[-1,:] == 1.0) # No Longer
    homogenous_coordinates = np.ones((1, points_3d.shape[1]))
    points_3d = np.vstack((points_3d, homogenous_coordinates))
    ##
    points_2d = P_Cam @ points_3d
    points_2d = points_2d / points_2d[-1,:] # Normalize by dividing by Homogenous Coordinate
    assert np.all(points_2d[-1,:] == 1.0)
    points_2d = points_2d[:-1,:]

    return points_2d

def AssertTracksShape(tracks):
    """Assert Tracks Shape. (Must contain equal number of samples)
       
    Args:
        tracks: 

    Returns:
        Number of Samples
    """    
    num_points = tracks.pts_2d_left.shape[1]
    assert (tracks.pts_2d_left.shape[0] == 2)
    assert (tracks.pts_2d_right.shape == (2, num_points))
    assert (tracks.pts_3d_mean.shape == (3, num_points))
    # assert np.all(tracks.pts_3d_mean[-1,:] == 1.0) # No Longer Stores Homogenous Coordinate
    assert (tracks.pts_3d_cov.shape  == (num_points, 3, 3))
    
    return num_points

# Based on Thrun Motion Model
def GetOdometryErrorParameters():
    """Odometry Error Parameters. Based on Motion Model from `Thrun et al Probabilistic Robotics`
       
    Args:

    Returns:
        Error Params
    """
    err_params = SimpleNamespace()
    # Translational error from translational speed
    err_params.alpha1 = 0.1 # 0.05 to 0.1 (Unitless) Represents a 5% to 10% error in distance traveled.
    # Translational error from rotational speed 
    # Models how much turning the wheels on a spot accidentally causes the robot to shift linearly.
    err_params.alpha2 = 0.01 # 0.001 to 0.01 (m ⋅ s / rad). 
    # Rotational error from translational speed: 
    # Models how much driving forward causes the robot to veer off course due to uneven wheel diameters or floor surfaces.
    err_params.alpha3 = 0.2 # 0.05 to 0.2 (rad / m)
    # Rotational error from rotational speed: 
    # Represents a 5% to 10% error in the execution of a pure turn.
    err_params.alpha4 = 0.1 # 0.05 to 0.1 (Unitless) 

    return err_params

# Based on Thrun Motion Model
def ComputeOdometryCovariance(vf, 
                              vr, 
                              wu, 
                              wl,
                              delta_seconds):
    """Compute Odometry Covariance Matrix as a function of the Odometry values.
    Based on Motion Model from `Thrun et al Probabilistic Robotics`
       
    Args:
        vf: Forward Velocity
        vr: Rightward Velocity 
        wu: Yaw Velocity (Up-Axis)
        wl: Pitch Velocity (Left-Axis)
        delta_seconds: seconds elapsed within this time-step

    Returns:
        Odometry Covariance Matrix
    """
    err = GetOdometryErrorParameters()
    Q_input_space = np.eye(4)
    # Variance in Lateral Displacement (X)
    Q_input_space[0,0] = 0.5 # (err.alpha1*(vr**2) + err.alpha2*(wu**2))*(delta_seconds**2) 
    # Variance in Longitudinal Displacement (Z)
    Q_input_space[1,1] = 0.5 # (err.alpha1*(vf**2) + err.alpha2*(wu**2))*(delta_seconds**2) 
    # Variance in Yaw Displacement (theta)
    Q_input_space[2,2] = 0.1 # (err.alpha3*(vf**2) + err.alpha4*(wu**2))*(delta_seconds**2)  
    # Variance in Pitch Displacement (theta)
    Q_input_space[3,3] = 0.05 # (err.alpha3*(vf**2) + err.alpha4*(wl**2))*(delta_seconds**2)  

    return Q_input_space

"""
See `vitekf_compute_jacobians.py` for details
--------------------------------------------------------------------------------------------------------------------
------------------------------------------ State Transition Model --------------------------------------------------- 
⎡1.0      0.0            0.0     ⎤ ⎡cos(δ_yaw)   0.0  sin(δ_yaw)⎤ ⎡ -δₓ + μₓ ⎤
⎢                                ⎥ ⎢                            ⎥ ⎢          ⎥
⎢0.0  cos(δ_pitch)  -sin(δ_pitch)⎥⋅⎢    0.0      1.0     0.0    ⎥⋅⎢   μ_y    ⎥
⎢                                ⎥ ⎢                            ⎥ ⎢          ⎥
⎣0.0  sin(δ_pitch)  cos(δ_pitch) ⎦ ⎣-sin(δ_yaw)  0.0  cos(δ_yaw)⎦ ⎣-δ_z + μ_z⎦

--------------------------------------------------------------------------------------------------------------------
------------ The Jacobian of the State Transition Model wrt "Control Inputs", `δ_x, δ_z, δ_yaw, δ_pitch`------------ 

⎡-1.0  0   -1.0⋅δ_z + 1.0⋅μ_z      0    ⎤
⎢                                       ⎥
⎢ 0    0           0           δ_z - μ_z⎥
⎢                                       ⎥
⎣ 0    -1       δₓ - μₓ         1.0⋅μ_y ⎦
--------------------------------------------------------------------------------------------------------------------
"""

def ComputeProcessNoise(Q_input_space,
                        pts_3d_mean,
                        delta_forward,
                        delta_right):
    """Compute Process Noise by projecting Odometry-Covariance into State-Space via Jacobian.
       
    Args:
        Q_input_space: Odometry-Covariance Matrix 
        pts_3d_mean: 3D Point Coordinate
        delta_forward: 
        delta_right:

    Returns:
        Process Noise Matrix.
    """
    assert pts_3d_mean.shape == (3,)
    x = pts_3d_mean[0]
    y = pts_3d_mean[1] 
    z = pts_3d_mean[2]
    # 
    G = np.array([[-1.0,  0.0, (z -delta_forward),        0.0         ],  
                  [ 0.0,  0.0,         0.0,        (delta_forward - z)],   
                  [ 0.0, -1.0,    (delta_right-x),      y             ]]) 
    Q_state_space = G @ Q_input_space @ G.T

    return Q_state_space

def Assert3DMeanShape(pts_3d_mean):
    assert pts_3d_mean.shape[0] == 3        # (x,y,z, 1.0)
    #assert np.all(pts_3d_mean[-1,:] == 1.0) # Is Homogenized
    num_points = pts_3d_mean.shape[1]

    return num_points


def ComputeDistributionsOf3DPoints(pts_3d_mean):
    """ Initialize Covariance Matrices of 3D Points as ellipsoids "extruded along" the Pixel 3D Line-of-Sight
    Depth variance as a function of depth: Variance(z) = z**4/(387.5744**2)*sigma_d**2
    where sigma_d = 0.5 or 1.0 (pixel matching variance)
       
    Args:
        points_3d: [4xN] Homogenous 3D coordinates

    Returns:
        pts_3d_cov [N, 3, 3]
    """
    num_points = Assert3DMeanShape(pts_3d_mean)
    pts_3d_x = pts_3d_mean[0,:].reshape(1,-1)
    pts_3d_y = pts_3d_mean[1,:].reshape(1,-1)
    pts_3d_z = pts_3d_mean[2,:].reshape(1,-1)
    # Line-of-Sight Angles
    los_yaw = np.atan(pts_3d_x/pts_3d_z)
    los_el  = np.atan(pts_3d_y/pts_3d_z)
    # 
    pix_match_var = 0.5
    pts_3d_z = pts_3d_mean[2,:].reshape(1,-1)
    z_variance = (pts_3d_z**4)/(387.5744**2)*pix_match_var**2/(np.cos(los_yaw)*np.cos(los_el))
    x_variance = (pts_3d_z/100.0)
    y_variance = x_variance
    pts_3d_var = np.concat((x_variance, 
                            y_variance, 
                            z_variance), axis=0) # [Var(x),Var(y),Var(z)]
    pts_3d_cov_unrot = np.eye(3) * pts_3d_var.T[:, :, np.newaxis]
    #
    R_y = np.zeros((num_points, 3, 3))
    R_x = np.zeros((num_points, 3, 3)) 
    for i_point in np.arange(num_points):
        yaw = los_yaw[0,i_point]
        el = los_el[0,i_point]
        R_y[i_point,:,:] = np.array([[ np.cos(yaw), 0.0, np.sin(yaw)],   # Clockwise
                                     [         0.0, 1.0,         0.0],   
                                     [-np.sin(yaw), 0.0, np.cos(yaw)]]) 
        R_x[i_point,:,:] = np.array([[1.0,         0.0,        0.0],   # Clockwise
                                     [0.0,  np.cos(el), np.sin(el)],   
                                     [0.0, -np.sin(el), np.cos(el)]]) 
        
    pts_3d_cov = R_x @ R_y @ pts_3d_cov_unrot @ np.transpose(R_y, (0, 2, 1)) @ np.transpose(R_x, (0, 2, 1))

    return pts_3d_cov
