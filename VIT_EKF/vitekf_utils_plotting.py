"""
    Plotting Utilities
    Supplements VIT-EKF by providing utilities for matplotlib data visualization.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import SVIT_EKF.vitekf_utils_core as utils
from types import SimpleNamespace
from scipy.stats import multivariate_normal

#                            X       Z
image_extent_vertical = [-40, 40,  0, 150] # m
#                            Z       Y 
image_extent_lateral  = [  0, 150,  5, -30]  # m
# 
track_scttr_size = 140
kDistributionVMax = 0.03
def ComputeGaussianCrossSectionalViews(tracks):
    """Compute Gaussian Cross Sectional Views
    Args:
        tracks: 
    Returns:
        gaussian_xsection Struct with Lateral & Vertical Images
    """
    num_points = utils.AssertTracksShape(tracks)
    image_size = 400  # pixels
    # Vertical View
    x = np.linspace(image_extent_vertical[0], image_extent_vertical[1], image_size)
    y = np.linspace(image_extent_vertical[2], image_extent_vertical[3], image_size)
    X, Y = np.meshgrid(x, y)
    pos_vert = np.dstack((X, Y))  # Stack grid arrays for multivariate normal evaluation
    # Lateral View
    x = np.linspace(image_extent_lateral[0], image_extent_lateral[1], image_size)
    y = np.linspace(image_extent_lateral[2], image_extent_lateral[3], image_size)
    X, Y = np.meshgrid(x, y)
    pos_lat = np.dstack((X, Y))  # Stack grid arrays for multivariate normal evaluation
    # 
    gaussian_xsection_vert = np.zeros((image_size, image_size))
    gaussian_xsection_lat  = np.zeros((image_size, image_size))
    for i_point in range(num_points):
        # Create the multivariate Gaussian distribution
        mean_xz = tracks.pts_3d_mean[(0,2),i_point]
        cov = tracks.pts_3d_cov[i_point,:,:] 
        cov_xz = cov[np.ix_([0,2],[0,2])]
        gaussian = multivariate_normal(mean=mean_xz, 
                                       cov=cov_xz, allow_singular=True)        
        z = gaussian.pdf(pos_vert)
        gaussian_xsection_vert += z
        # 
        mean_zy = tracks.pts_3d_mean[(2,1),i_point]
        cov_zy = cov[np.ix_([2,1],[2,1])]
        gaussian = multivariate_normal(mean=mean_zy, 
                                       cov=cov_zy, allow_singular=True)        
        z = gaussian.pdf(pos_lat)
        gaussian_xsection_lat += z
    # 
    gaussian_xsection = SimpleNamespace()
    gaussian_xsection.vert = gaussian_xsection_vert
    gaussian_xsection.lat  = gaussian_xsection_lat

    return gaussian_xsection

def InitializeVisualization(images, tracks):
    """Initialize Visualization
    Args:
        images:
        tracks: 
    Returns:
        plot_handles:
    """
    num_points = utils.AssertTracksShape(tracks)
    gaussian_xsection = ComputeGaussianCrossSectionalViews(tracks)
    # Init Visualization
    layout = [
        ['left1', 'left1', 'middle1', 'right1'],
        ['left1', 'left1', 'middle1', 'right1'],
        ['left2', 'left2', 'middle1', 'right1'],
        ['left2', 'left2', 'middle1', 'right1'],
        ['left3', 'left3', 'middle2', 'right2'],
        ['left3', 'left3', 'middle2', 'right2']
    ]
    # 19.2, 10.8 --> 33.6, 18.9
    fig, axd = plt.subplot_mosaic(layout, layout="constrained", figsize=(33.6, 18.9), dpi=100)
    fig.suptitle('Stereo Visual-Inertial Object Tracking with I-EKF', fontsize=60, weight='bold')
    title_font_size = 32
    label_font_size = 25
    # -----------------------------------------------------------------------------
    # Left 1 - Left Image 
    img_left_imshow  = axd['left1'].imshow(images.left, cmap='gray', vmin=0, vmax=255)
    pts_left_scatter = axd['left1'].scatter(tracks.pts_2d_left[0,:], 
                                            tracks.pts_2d_left[1,:], 
                                            color='cyan', s=80, label='Observation')
    reproj_left  = utils.ProjectPoints(tracks.pts_3d_mean, utils.P_Left)
    reproj_left_scatter = axd['left1'].scatter(reproj_left[0,:], 
                                                   reproj_left[1,:], 
                                                   edgecolors='lime', facecolors='none', linewidths=3.5, s=80, 
                                                   label='Corr Reproj')
    reproj_prefit_left_scatter = axd['left1'].scatter(reproj_left[0,:], 
                                                 reproj_left[1,:], 
                                                 edgecolors='red', facecolors='none', linewidths=2.0, s=80, 
                                                 label='Pred Reproj')
    axd['left1'].legend(loc='upper right', fontsize=16)    
    axd['left1'].set_title('Left Camera', 
                           fontsize=title_font_size, weight='bold')    
    axd['left1'].set_xticks([])
    axd['left1'].set_yticks([])
    # -----------------------------------------------------------------------------
    # Left 2 - Right Image 
    img_right_imshow  = axd['left2'].imshow(images.right, cmap='gray', vmin=0, vmax=255)
    pts_right_scatter = axd['left2'].scatter(tracks.pts_2d_right[0,:], 
                                            tracks.pts_2d_right[1,:], 
                                            color='cyan', s=80, label='Observation')
    reproj_right  = utils.ProjectPoints(tracks.pts_3d_mean, utils.P_Right)
    reproj_right_scatter = axd['left2'].scatter(reproj_right[0,:], 
                                                reproj_right[1,:], 
                                                edgecolors='lime', facecolors='none', linewidths=3.5, s=80, 
                                                label='Corr Reproj')
    reproj_prefit_right_scatter = axd['left2'].scatter(reproj_right[0,:], 
                                                       reproj_right[1,:], 
                                                       edgecolors='red', facecolors='none', linewidths=2.0, s=80, 
                                                       label='Pred Reproj')
    #axd['left2'].legend(loc='upper right', fontsize=16)    
    axd['left2'].set_title('Right Camera', fontsize=title_font_size, weight='bold')    
    axd['left2'].set_xticks([])
    axd['left2'].set_yticks([])
    # Left 2
    # Reprojection Errors (Residuals)
    post_fit_res = np.sum((tracks.pts_2d_left - reproj_left)**2, axis=0)**(0.5)
    post_fit_res_left_plot,  = axd['left3'].plot(np.arange(num_points), post_fit_res, 
                                                 color='green', label='Corrected Left', linewidth=4.0)
    pre_fit_res_left_plot,   = axd['left3'].plot(np.arange(num_points), post_fit_res, 
                                                 color='red',  label='Predicted Left', linewidth=3.0)
    post_fit_res_right_plot,  = axd['left3'].plot(np.arange(num_points), post_fit_res, 
                                                  color='green', label='Corrected Right', linewidth=4.0, linestyle='dashed')
    pre_fit_res_right_plot,   = axd['left3'].plot(np.arange(num_points), post_fit_res, 
                                                  color='red',  label='Predicted Right', linewidth=3.0, linestyle='dashed')    
    axd['left3'].set_title('Residuals: Reprojection Errors', fontsize=title_font_size, weight='bold')
    axd['left3'].set_xlabel('Track Index', fontsize=label_font_size, weight='bold')
    axd['left3'].set_ylabel('Distance (pixels)', fontsize=label_font_size, weight='bold')
    axd['left3'].legend(loc='upper right', fontsize=18)  
    axd['left3'].grid(True)
    axd['left3'].set_ylim(-2,15)
    # ----------------------------------------------------------------------------------------
    # Middle 1 PREDICT
    vert_pred_scatter = axd['middle1'].scatter(tracks.pts_3d_mean[0,:], 
                                                 tracks.pts_3d_mean[2,:], 
                                                 facecolors='violet',   
                                                 edgecolors='black',   
                                                 linewidths=2.5,        
                                                 s=track_scttr_size)
    vert_pred_imshow  = axd['middle1'].imshow(gaussian_xsection.vert, 
                                                cmap='magma', origin='lower', 
                                                extent=image_extent_vertical, vmin=0, vmax=kDistributionVMax)
    axd['middle1'].set_xlabel('X Lateral Axis (m)', fontsize=label_font_size, weight='bold')
    axd['middle1'].set_ylabel('Vertical View', fontsize=title_font_size, weight='bold')
    axd['middle1'].set_title('Predicted', fontsize=title_font_size+5, weight='bold')
    axd['middle1'].grid(True, color='violet', linestyle='--', linewidth=0.5)
    axd['middle1'].set_aspect("equal")
    # Middle 2 PREDICT
    lateral_pred_scatter = axd['middle2'].scatter(tracks.pts_3d_mean[2,:], 
                                              tracks.pts_3d_mean[1,:], 
                                              facecolors='violet',   
                                              edgecolors='black',   
                                              linewidths=2.5,        
                                              s=track_scttr_size)                 
    lateral_pred_imshow = axd['middle2'].imshow(gaussian_xsection.lat, 
                                            cmap='magma', origin='lower', 
                                            extent=image_extent_lateral, vmin=0, vmax=kDistributionVMax)
    axd['middle2'].set_ylim((image_extent_lateral[2], image_extent_lateral[3]))
    axd['middle2'].set_xlabel('Z Longitudinal Axis (m)', fontsize=label_font_size, weight='bold')
    axd['middle2'].set_ylabel('Side View', fontsize=title_font_size, weight='bold')
    axd['middle2'].grid(True, color='violet', linestyle='--', linewidth=0.5)
    # ----------------------------------------------------------------------------------------
    # Right 1 CORRECT
    vert_corr_scatter = axd['right1'].scatter(tracks.pts_3d_mean[0,:], 
                                              tracks.pts_3d_mean[2,:], 
                                              facecolors='violet',   
                                              edgecolors='black',   
                                              linewidths=2.5,        
                                              s=track_scttr_size)
    vert_corr_imshow  = axd['right1'].imshow(gaussian_xsection.vert, 
                                             cmap='magma', origin='lower', 
                                             extent=image_extent_vertical, vmin=0, vmax=kDistributionVMax)
    axd['right1'].set_xlabel('X Lateral Axis (m)', fontsize=label_font_size, weight='bold')
    axd['right1'].set_ylabel('Z Longitudinal Axis (m)', fontsize=label_font_size, weight='bold')
    axd['right1'].set_title('Corrected', fontsize=title_font_size+5, weight='bold')
    axd['right1'].grid(True, color='violet', linestyle='--', linewidth=0.5)
    axd['right1'].set_aspect("equal")
    # Right 2 CORRECT
    lateral_corr_scatter = axd['right2'].scatter(tracks.pts_3d_mean[2,:], 
                                                 tracks.pts_3d_mean[1,:], 
                                                 facecolors='violet',   
                                                 edgecolors='black',   
                                                 linewidths=2.5,        
                                                 s=track_scttr_size)                 
    lateral_corr_imshow = axd['right2'].imshow(gaussian_xsection.lat, 
                                               cmap='magma', origin='lower', 
                                               extent=image_extent_lateral, vmin=0, vmax=kDistributionVMax)
    axd['right2'].set_ylim((image_extent_lateral[2], image_extent_lateral[3]))
    axd['right2'].set_xlabel('Z Longitudinal Axis (m)', fontsize=label_font_size, weight='bold')
    axd['right2'].set_ylabel('Y Vertical Axis (m)', fontsize=label_font_size, weight='bold')
    axd['right2'].grid(True, color='violet', linestyle='--', linewidth=0.5)
    ## ===================================================================================
    plot_handles = SimpleNamespace()
    plot_handles.fig = fig
    plot_handles.axd = axd
    # Left 1
    plot_handles.img_left_imshow = img_left_imshow
    plot_handles.pts_left_scatter = pts_left_scatter
    plot_handles.reproj_left_scatter = reproj_left_scatter
    plot_handles.reproj_prefit_left_scatter = reproj_prefit_left_scatter
    # Left 2
    plot_handles.img_right_imshow = img_right_imshow
    plot_handles.pts_right_scatter = pts_right_scatter
    plot_handles.reproj_right_scatter = reproj_right_scatter
    plot_handles.reproj_prefit_right_scatter = reproj_prefit_right_scatter
    # Left 3 
    plot_handles.post_fit_res_left_plot  = post_fit_res_left_plot 
    plot_handles.pre_fit_res_left_plot   = pre_fit_res_left_plot
    plot_handles.post_fit_res_right_plot  = post_fit_res_right_plot 
    plot_handles.pre_fit_res_right_plot   = pre_fit_res_right_plot    
    # Middle 1
    plot_handles.vert_pred_scatter = vert_pred_scatter
    plot_handles.vert_pred_imshow = vert_pred_imshow                      
    # Middle 2
    plot_handles.lateral_pred_scatter = lateral_pred_scatter
    plot_handles.lateral_pred_imshow = lateral_pred_imshow
    # Right 1
    plot_handles.vert_corr_scatter = vert_corr_scatter
    plot_handles.vert_corr_imshow = vert_corr_imshow                      
    # Right 2
    plot_handles.lateral_corr_scatter = lateral_corr_scatter
    plot_handles.lateral_corr_imshow = lateral_corr_imshow
    
    return plot_handles
    
def UpdatePredictedXSectionViews(plot_handles,
                                 tracks):
    """Update Prediction Tracks Cross-Sectional Views
    Args:
        plot_handles:
        tracks: 
    Returns:
    """
    gaussian_xsection = ComputeGaussianCrossSectionalViews(tracks)
    # axd['middle2']
    plot_handles.lateral_pred_scatter.set_offsets(np.vstack((tracks.pts_3d_mean[2,:], tracks.pts_3d_mean[1,:])).T)
    plot_handles.lateral_pred_imshow.set_data(gaussian_xsection.lat) # Y-Axis Points Downward
    # axd['middle1']
    plot_handles.vert_pred_scatter.set_offsets(np.vstack((tracks.pts_3d_mean[0,:], tracks.pts_3d_mean[2,:])).T)
    plot_handles.vert_pred_imshow.set_data(gaussian_xsection.vert)

    return

def UpdateCorrectedXSectionViews(plot_handles,
                                 tracks):
    """Update Corrected Tracks Cross-Sectional Views
    Args:
        plot_handles:
        tracks: 
    Returns:
    """
    gaussian_xsection = ComputeGaussianCrossSectionalViews(tracks)
    # axd['middle2']
    plot_handles.lateral_corr_scatter.set_offsets(np.vstack((tracks.pts_3d_mean[2,:], tracks.pts_3d_mean[1,:])).T)
    plot_handles.lateral_corr_imshow.set_data(gaussian_xsection.lat) # Y-Axis Points Downward
    # axd['middle1']
    plot_handles.vert_corr_scatter.set_offsets(np.vstack((tracks.pts_3d_mean[0,:], tracks.pts_3d_mean[2,:])).T)
    plot_handles.vert_corr_imshow.set_data(gaussian_xsection.vert)

    return

def UpdateImageAndReprojectionError(plot_handles, 
                                    images_next, 
                                    tracks,
                                    pts_3d_mean_pred):
    """Update Image, Reprojected Points, and Reprojection Error Plots
    Args:
        plot_handles:
        images_next:
        tracks: 
    Returns:
    """
    num_points = utils.AssertTracksShape(tracks)
    # print('reproj_pred.shape ', reproj_pred.shape)
    # print('num_points ', num_points)
    assert pts_3d_mean_pred.shape == (3, num_points)
    #assert np.all(pts_3d_mean_pred[-1,:] == 1.0) # Is Homogenized
    # Predicted Reprojections Left/Right
    reproj_left_pred   = utils.ProjectPoints(pts_3d_mean_pred, utils.P_Left)  # Only Left is used in KF
    reproj_right_pred  = utils.ProjectPoints(pts_3d_mean_pred, utils.P_Right) # Only Left is used in KF
    # Measurement-Corrected Reprojections Left/Right
    reproj_left   = utils.ProjectPoints(tracks.pts_3d_mean, utils.P_Left)   # Only Left is used in KF
    reproj_right  = utils.ProjectPoints(tracks.pts_3d_mean, utils.P_Right)  # Only Left is used in KF
    # Residuals
    pre_fit_res_left   = np.sum((tracks.pts_2d_left  - reproj_left_pred)**2, axis=0)**(0.5)  # Pre-Fit  Left
    post_fit_res_left  = np.sum((tracks.pts_2d_left  - reproj_left)**2, axis=0)**(0.5)       # Post-Fit Left
    pre_fit_res_right  = np.sum((tracks.pts_2d_right - reproj_right_pred)**2, axis=0)**(0.5) # Pre-Fit  Right
    post_fit_res_right = np.sum((tracks.pts_2d_right - reproj_right)**2, axis=0)**(0.5)      # Post-Fit Right
    # Left 1
    plot_handles.img_left_imshow.set_data(images_next.left)
    plot_handles.pts_left_scatter.set_offsets(tracks.pts_2d_left.T)
    plot_handles.reproj_left_scatter.set_offsets(reproj_left.T)
    plot_handles.reproj_prefit_left_scatter.set_offsets(reproj_left_pred.T)
    # Left 2
    plot_handles.img_right_imshow.set_data(images_next.right)
    plot_handles.pts_right_scatter.set_offsets(tracks.pts_2d_right.T)
    plot_handles.reproj_right_scatter.set_offsets(reproj_right.T)
    plot_handles.reproj_prefit_right_scatter.set_offsets(reproj_right_pred.T)
    # ------- Left 3 - Residuals -------   
    x_values = np.arange(num_points)
    # Left Residuals
    plot_handles.post_fit_res_left_plot.set_xdata(x_values) 
    plot_handles.post_fit_res_left_plot.set_ydata(post_fit_res_left)
    plot_handles.pre_fit_res_left_plot.set_xdata(x_values) 
    plot_handles.pre_fit_res_left_plot.set_ydata(pre_fit_res_left)
    # Right Residuals
    plot_handles.post_fit_res_right_plot.set_xdata(x_values) 
    plot_handles.post_fit_res_right_plot.set_ydata(post_fit_res_right)
    plot_handles.pre_fit_res_right_plot.set_xdata(x_values) 
    plot_handles.pre_fit_res_right_plot.set_ydata(pre_fit_res_right)
    plot_handles.axd['left3'].set_xlim(0, num_points)     

    return

def InitializeVideoWriter(fig, file_name):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 10
    #width, height = fig.canvas.get_width_height()
    img_rgba = np.asarray(fig.canvas.buffer_rgba())
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    # 4. Extract the EXACT physical pixel dimensions from the shape
    # NumPy shape order is (Height, Width, Channels)
    true_height, true_width, _ = img_bgr.shape
    writer = cv2.VideoWriter(file_name, 
                             fourcc, 
                             fps, 
                             (true_width, true_height)) 
    
    return writer

def WriteToVideo(writer, fig):
    img_rgba = np.asarray(fig.canvas.buffer_rgba())
    # OpenCV requires BGR format, drop the Alpha channel and reverse RGB -> BGR
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    # Write the frame to the video file
    writer.write(img_bgr)
    return
