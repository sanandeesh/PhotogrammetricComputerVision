"""
    VIT-EKF Core
    Core VIT-EKF Functions for 
    (1) Initializing Tracks, 
    (2) Updating Tracks, 
    (3) KF Prediction-Step, 
    (4) KF Measurement-Correction Step
"""
import numpy as np
import cv2
import SVIT_EKF.vitekf_utils_core as utils
from types import SimpleNamespace


def InitializeTracks(images):
    """ Initialize Tracks given Image-Pair. Tracks consist of 
    1. 2D Features Tracked
    2. 3D Triangulated Features Tracked
    3. 3x3 Covariance Matrix of each Tracked Point
    Args:
        images: Left & Right Image
    Returns:
        tracks Struct
    """
    # [N x 1 x 2]
    # EXPERIMENT
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_img_nxt_lft = clahe.apply(images.left)
    #
    pts_2d_left_cv = cv2.goodFeaturesToTrack(equalized_img_nxt_lft, #images.left, 
                                             mask = utils.GetFeatureROIMask(images.left), 
                                             **utils.feature_params) 
    pts_2d_left = np.squeeze(pts_2d_left_cv, axis=1).T # 2xN
    pts_2d_right = utils.ComputePointCorrespondences(images.left, 
                                                     images.right, 
                                                     pts_2d_left.astype(int), 
                                                     utils.kStereoMatchWindowSize)
    pts_3d_mean, pts_2d_left, pts_2d_right = utils.TriangulateFeaturesRemoveInvalidPoints(pts_2d_left, 
                                                                                          pts_2d_right,
                                                                                          utils.kHomogenousThreshold)
    pts_3d_cov = utils.ComputeDistributionsOf3DPoints(pts_3d_mean)
    # TRACKS INITIALIZATION
    tracks = SimpleNamespace()
    tracks.pts_2d_left  = pts_2d_left  # N Observations [2,N]
    tracks.pts_2d_right = pts_2d_right # N Observations [2,N]
    tracks.pts_3d_mean  = pts_3d_mean  # N Exp States   [4,N] (inc Homog Coord = 1.0)
    tracks.pts_3d_cov   = pts_3d_cov   # N Covariances  [N,3,3] 
    #
    num_points = utils.AssertTracksShape(tracks)

    return tracks

def RemoveNegativeTracks(tracks_next):
    positive_tracks = ((tracks_next.pts_2d_left[0,:] >= 0) & 
                       (tracks_next.pts_2d_left[1,:] >= 0) &
                       (tracks_next.pts_3d_mean[2,:] >= 0))
    tracks_next.pts_2d_left = tracks_next.pts_2d_left[:,positive_tracks]
    tracks_next.pts_3d_mean = tracks_next.pts_3d_mean[:,positive_tracks]
    tracks_next.pts_3d_cov  = tracks_next.pts_3d_cov[positive_tracks,:,:]

    return tracks_next

def TrackFeaturesAndReplaceLostPoints(images,
                                      tracks,
                                      images_next):
    """ 
    Track features across two images, and replace lost points (e.g. left FoV)
    Args:
        images: Left & Right Image
        tracks: 
        images_next: images of next time-step
    Returns:
        images: Updated to `images_next`
        tracks: Updated
    """
    num_points = utils.AssertTracksShape(tracks)
    pts_2d_left_cv = tracks.pts_2d_left.T.reshape(-1,1,2) # [N,1,2]
    pts_2d_left_next_cv, st, err = cv2.calcOpticalFlowPyrLK(images.left, 
                                                            images_next.left, 
                                                            pts_2d_left_cv, 
                                                            None, 
                                                            **utils.lk_params)
    # Remove Invalid Tracks (e.g. left FoV)
    tracks_next = SimpleNamespace()
    tracks_next.pts_2d_left  = pts_2d_left_next_cv[st==1].T                 # [2,N]
    tracks_next.pts_3d_mean  = tracks.pts_3d_mean[:,(st.T == 1).squeeze()]  # [4,N]
    tracks_next.pts_3d_cov   = tracks.pts_3d_cov[(st.T == 1).squeeze(),:,:] # [3,3,N]
    tracks_next = RemoveNegativeTracks(tracks_next)
    tracks_next.pts_2d_right = utils.ComputePointCorrespondences(images_next.left, 
                                                                 images_next.right, 
                                                                 tracks_next.pts_2d_left.astype(int), 
                                                                 utils.kStereoMatchWindowSize)
    num_points_next = utils.AssertTracksShape(tracks_next)
    # Replace points lost outside the FoV
    num_lost_points = num_points - num_points_next
    if (num_points_next < utils.kMaxCorners): # num_lost_points:
        # EXPERIMENT
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_img_nxt_lft = clahe.apply(images_next.left)
        #
        utils.feature_params['maxCorners'] = utils.kMaxCorners - num_points_next # num_lost_points 
        pts_2d_left_new_cv = cv2.goodFeaturesToTrack(equalized_img_nxt_lft, #images_next.left, 
                                                     mask = utils.GetFeatureROIMask_Dynamic(images_next.left, 
                                                                                            tracks_next.pts_2d_left), 
                                                     **utils.feature_params)
        if (pts_2d_left_new_cv is not None):
            pts_2d_left_new  = np.squeeze(pts_2d_left_new_cv, axis=1).T
            # 
            pts_2d_right_new = utils.ComputePointCorrespondences(images_next.left, 
                                                                        images_next.right, 
                                                                        pts_2d_left_new.astype(int), 
                                                                        utils.kStereoMatchWindowSize)
            pts_3d_mean_new, pts_2d_left_new, pts_2d_right_new = utils.TriangulateFeaturesRemoveInvalidPoints(pts_2d_left_new, 
                                                                                                              pts_2d_right_new, 
                                                                                                              utils.kHomogenousThreshold)
            pts_3d_cov_new = utils.ComputeDistributionsOf3DPoints(pts_3d_mean_new)
            tracks_next.pts_2d_left  = np.concatenate((tracks_next.pts_2d_left,  pts_2d_left_new),  axis=1) # Horiz Concat
            tracks_next.pts_2d_right = np.concatenate((tracks_next.pts_2d_right, pts_2d_right_new), axis=1) # Horiz Concat
            tracks_next.pts_3d_mean  = np.concatenate((tracks_next.pts_3d_mean,  pts_3d_mean_new),  axis=1) # Horiz Concat
            tracks_next.pts_3d_cov   = np.concatenate((tracks_next.pts_3d_cov,   pts_3d_cov_new),   axis=0) # Stack Concat
            #
            num_points_next = utils.AssertTracksShape(tracks_next)

    return tracks_next

# Points Move in Ego-Frame of Moving Car, so apply the "Opposite" motion to Points
def ForwardPropagate3DPoints(index, oxts_data, tracks):
    """ Prediction Step: apply Forward\Lateral translation, and Yaw\Pitch rotation from KITTI OXTS to 3D Tracks.
    Args:
        index: timestep index to grab corresponding OXTS inertial data
        oxts_data: 
        tracks:
    Returns:
        Forward Propagated `tracks`
    """
    num_points = utils.AssertTracksShape(tracks)
    delta_seconds = oxts_data.local_timestamps[index] - oxts_data.local_timestamps[index-1]
    v_forward =  (oxts_data.vf[index-1] + oxts_data.vf[index])/2.0 
    v_right   = -(oxts_data.vl[index-1] + oxts_data.vl[index])/2.0 # Switch from Leftward to Rightward Velocity
    w_up      =  (oxts_data.wu[index-1] + oxts_data.wu[index])/2.0 # Angular Velocity about vertical Axis (CCW)
    w_left    =  (oxts_data.wl[index-1] + oxts_data.wl[index])/2.0 # Angular Velocity about Left Axis (?)
    delta_forward = v_forward*delta_seconds
    delta_right   = v_right*delta_seconds
    delta_yaw     = w_up*delta_seconds    # CounterClockwise
    delta_pitch   = w_left*delta_seconds    # ?
    #
    R_y = np.array([[np.cos(delta_yaw), 0.0,  np.sin(delta_yaw)],   # Clockwise
                    [             0.0,  1.0,                0.0],   
                    [-np.sin(delta_yaw), 0.0, np.cos(delta_yaw)]]) 
    R_x = np.array([[1.0,                 0.0,                   0.0],
                    [0.0, np.cos(delta_pitch), -np.sin(delta_pitch)],
                    [0.0, np.sin(delta_pitch),  np.cos(delta_pitch)]])    
    shift = np.array([[-delta_right],   # X Axis Rightward +
                               [0.0],   # Y Axis Downward +
                     [-delta_forward]]) # Z Axis Forward +
    # Predict Mean
    pts_3d_mean_pred = R_x @ R_y @ (tracks.pts_3d_mean + shift) # Ignore the Homogenous Coordinates
    #pts_3d_mean_pred = np.concat((pts_3d_mean_pred, np.ones((1,num_points))), axis=0)
    # Predict Covariance
    Q_input_space = utils.ComputeOdometryCovariance(v_forward,
                                                    v_right,
                                                    w_up,
                                                    w_left,
                                                    delta_seconds)
    Q_state_space = np.zeros(tracks.pts_3d_cov.shape)
    for i_point in range(num_points):
        Q_state_space[i_point,:,:] = utils.ComputeProcessNoise(Q_input_space, 
                                                               tracks.pts_3d_mean[:,i_point],
                                                               delta_forward,
                                                               delta_right)
    Q = Q_state_space
    pts_3d_cov_pred = R_x @ R_y @ tracks.pts_3d_cov @ R_y.T @ R_x.T + Q 
    # Update Tracks 
    tracks.pts_3d_mean = pts_3d_mean_pred
    tracks.pts_3d_cov  = pts_3d_cov_pred

    return tracks


"""
See `vitekf_compute_jacobians.py` for details
--------------------------------------------------------------------------------------------------------------------
------------------------------------------ Measurement Model Jacobian ---------------------------------------------- 
Measurement Model: Non-Linear Normalized Camera Projection
     ⎡fₓ        cₓ ⎤      
     ⎢───  0.0  ───⎥ ⎡μₓ ⎤
     ⎢μ_z       μ_z⎥ ⎢   ⎥
 h = ⎢             ⎥⋅⎢μ_y⎥
     ⎢     f_y  c_y⎥ ⎢   ⎥
     ⎢0.0  ───  ───⎥ ⎣μ_z⎦
     ⎣     μ_z  μ_z⎦      
--------------------------------------------------
Jacobian wrt State (μ_x μ_y,μ_z):
     ⎡ X         -fₓ⋅X   ⎤
     ⎢───   0    ─────── ⎥
     ⎢ Z             2   ⎥
     ⎢              Z    ⎥
 H = ⎢                   ⎥
     ⎢      Y   -f_y⋅Y   ⎥
     ⎢ 0   ───  ─────────⎥
     ⎢      Z        2   ⎥
     ⎣              Z    ⎦
--------------------------------------------------------------------------------------------------------------------
"""
def ComputeMeasurementModelJacobian(mean_predicted, P_cam):
    """ Compute Measurement Model Jacobian
    Args:
        mean_predicted: 

    Returns:
        Jacobian of Non-Linear Projection Model of Left Camera
    """
    assert (mean_predicted.shape == (3,1))
    X = mean_predicted[0,0] 
    Y = mean_predicted[1,0] 
    Z = mean_predicted[2,0] 
    fx = P_cam[0,0]
    fy = P_cam[1,1]
    bx = P_cam[0,3]
    H = np.array([
        [fx / Z,   0.0,    -((fx * X) / (Z**2))-(bx/(Z**2))],
        [0.0,    fy / Z,   -(fy * Y) / (Z**2)]
    ])

    return H

def ComputeMeasurementCorrection(mean_pred, cov_pred, P_cam, measurement):
    """ Measurement Correction Step: Update Predicted Tracks with Measurement Error time Kalman Gain
    Args:
        mean_pred: 
        cov_pred:
        P_cam:
        measurement_error:
    Returns:
        Measurement Corrected Distribution
    """
    assert ((mean_pred.shape == (3,1)) and 
            (cov_pred.shape == (3,3))   and 
            (measurement.shape == (2,1)) and
            (P_cam.shape == (3,4)))
    #
    reprojection  = utils.ProjectPoints(mean_pred, P_cam)
    measurement_error = measurement - reprojection
    H = ComputeMeasurementModelJacobian(mean_pred, P_cam)
    S = (H @ cov_pred @ H.T) + utils.R
    S_inv = np.linalg.inv(S)
    KalmanGain = (cov_pred @ H.T) @ S_inv
    mean_corrected = mean_pred + KalmanGain @ measurement_error # residual
    i_minus_KH = np.eye(3) - KalmanGain @ H
    # Joseph Form
    cov_corrected = (i_minus_KH @ cov_pred @ i_minus_KH.T) + (KalmanGain @ utils.R @ KalmanGain.T)
    # cov_corrected = i_minus_KH @ cov_pred

    return mean_corrected, cov_corrected

def ComputeStereoMeasurementCorrection(mean_pred, cov_pred, measurement_l, measurement_r):
    """ Measurement Correction Step: Update Predicted Tracks with Measurement Error time Kalman Gain
    Args:
        mean_pred: 
        cov_pred:
        measurement_l:
        measurement_r:
    Returns:
        Measurement Corrected Distribution
    """
    assert ((mean_pred.shape == (3,1)) and 
            (cov_pred.shape == (3,3))   and 
            (measurement_l.shape == (2,1)) and
            (measurement_r.shape == (2,1)))
    #
    measurement = np.vstack((measurement_l, measurement_r))
    reprojection_l  = utils.ProjectPoints(mean_pred, utils.P_Left)
    reprojection_r  = utils.ProjectPoints(mean_pred, utils.P_Right)
    reprojection = np.vstack((reprojection_l, reprojection_r))
    measurement_error = measurement - reprojection
    H_l = ComputeMeasurementModelJacobian(mean_pred, utils.P_Left)
    H_r = ComputeMeasurementModelJacobian(mean_pred, utils.P_Right)
    H = np.vstack((H_l, H_r))
    S = (H @ cov_pred @ H.T) + utils.R_stereo
    S_inv = np.linalg.inv(S)
    KalmanGain = (cov_pred @ H.T) @ S_inv
    mean_corrected = mean_pred + KalmanGain @ measurement_error # residual
    i_minus_KH = np.eye(3) - KalmanGain @ H
    # Joseph Form
    cov_corrected = (i_minus_KH @ cov_pred @ i_minus_KH.T) + (KalmanGain @ utils.R_stereo @ KalmanGain.T)
    # cov_corrected = i_minus_KH @ cov_pred

    return mean_corrected, cov_corrected

def ComputeStereoIteratedMeasurementCorrection(mean_pred, cov_pred, measurement_l, measurement_r):
    """ Measurement Correction Step: Update Predicted Tracks with Measurement Error time Kalman Gain
    Args:
        mean_pred: 
        cov_pred:
        measurement_left:
        measurement_right:
    Returns:
        mean_corrected, cov_corrected 
    """
    assert ((mean_pred.shape == (3,1)) and 
            (cov_pred.shape == (3,3))   and 
            (measurement_l.shape == (2,1)) and
            (measurement_r.shape == (2,1)))
    #
    mean_corrected = mean_pred.copy()
    for i_iter in range(utils.kNumCorrectionCycles):
        reprojection_l  = utils.ProjectPoints(mean_corrected, utils.P_Left)
        reprojection_r  = utils.ProjectPoints(mean_corrected, utils.P_Right)
        # Break Early if Residual-Error Sufficiently Low
        error_mag_l   = np.sum((measurement_l  - reprojection_l)**2, axis=0)**(0.5)
        error_mag_r   = np.sum((measurement_r  - reprojection_r)**2, axis=0)**(0.5)
        if ((i_iter > 0) and # Measurement-Correction Runs AT LEAST ONCE
            (error_mag_l < utils.kResidualThreshold) and 
            (error_mag_r < utils.kResidualThreshold)):
            break 
        #
        reprojection = np.vstack((reprojection_l, reprojection_r))
        measurement  = np.vstack((measurement_l, measurement_r))
        measurement_error = measurement - reprojection #- H @ (mean_pred - mean_corrected)
        #
        H_l = ComputeMeasurementModelJacobian(mean_corrected, utils.P_Left)
        H_r = ComputeMeasurementModelJacobian(mean_corrected, utils.P_Right)
        H = np.vstack((H_l, H_r))
        S = (H @ cov_pred @ H.T) + utils.R_stereo
        S_inv = np.linalg.inv(S)
        KalmanGain = (cov_pred @ H.T) @ S_inv
        mean_corrected = mean_corrected + KalmanGain @ measurement_error # residual

    # Update Covariance
    i_minus_KH = np.eye(3) - KalmanGain @ H
    # Joseph Form
    cov_corrected = (i_minus_KH @ cov_pred @ i_minus_KH.T) + (KalmanGain @ utils.R_stereo @ KalmanGain.T)

    return mean_corrected, cov_corrected

def KFMeasurementCorrectionStep(tracks):
    """ Measurement Correction Step: Update Predicted Tracks with Measurement Error time Kalman Gain
    Args:
        tracks: 
    Returns:
        Updated tracks
    """    
    num_points = utils.AssertTracksShape(tracks)
    for i_point in range(num_points):
        mean        = tracks.pts_3d_mean[:,i_point].copy().reshape(3,1)
        covariance  = tracks.pts_3d_cov[i_point,:,:].copy().reshape(3,3)
        measurement_left = tracks.pts_2d_left[:,i_point].reshape(2,1)
        measurement_right = tracks.pts_2d_right[:,i_point].reshape(2,1)
        # ------------------------------------------------------------------------
        mean, covariance = ComputeStereoIteratedMeasurementCorrection(mean, 
                                                                      covariance, 
                                                                      measurement_left, 
                                                                      measurement_right)
        # ------------------------------------------------------------------------
        tracks.pts_3d_mean[:,i_point]  = mean.reshape(3,)
        tracks.pts_3d_cov[i_point,:,:] = covariance

    return tracks
