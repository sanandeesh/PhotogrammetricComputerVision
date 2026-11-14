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
from matplotlib.patches import Ellipse
from scipy.stats import chi2
 
kZMax = 150.0
#                            X       Z
image_extent_vertical = [-40, 40,  0, kZMax] # m
#                             Z         Y 
image_extent_lateral  = [  0, kZMax,  5, -30]  # m
# 
kTrackScttrSize = 140
kDistributionVMax = 0.03

kPredictedColor = (112/255, 128/255, 144/255, 0.05) 
kCorrectedColor = (112/255, 128/255, 144/255, 1.0) 
chi2_90 = chi2.ppf(0.90, df=2)
kfps = 10 # Video Recording FPS


def ComputeCovarianceEllipse(mean, cov, color):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * np.sqrt(chi2_90 * eigvals)

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, facecolor=color, 
                      edgecolor='black', linewidth=0.5)

    return ellipse

def ComputeCovarianceEllipses(tracks, color = 'slategray'):
    vertical_view = []
    lateral_view  = []
    num_points = utils.AssertTracksShape(tracks)
    for i_point in range(num_points):
        mean = tracks.pts_3d_mean[:,i_point]
        if (mean[2] > kZMax):
            continue
        cov = tracks.pts_3d_cov[i_point,:,:]
        # Vertical View
        mean_xz = mean[(0,2),]
        cov_xz = cov[np.ix_([0,2],[0,2])]
        ellipse = ComputeCovarianceEllipse(mean_xz, cov_xz, color)
        vertical_view.append(ellipse)
        # Lateral View
        mean_zy = mean[(2,1),]
        cov_zy = cov[np.ix_([2,1],[2,1])]
        ellipse = ComputeCovarianceEllipse(mean_zy, cov_zy, color)
        lateral_view.append(ellipse)
    #
    covariance_ellipses = SimpleNamespace()
    covariance_ellipses.vertical_view = vertical_view
    covariance_ellipses.lateral_view  = lateral_view    
   
    return covariance_ellipses

def AddCovarianceEllipses(axis, covariance_ellipses):
    for patch in list(axis.patches):
        patch.remove()

    for ellipse in covariance_ellipses:
        axis.add_patch(ellipse)

    return

def InitializeVisualization(images, tracks):
    """Initialize Visualization
    Args:
        images:
        tracks: 
    Returns:
        plot_handles:
    """
    num_points = utils.AssertTracksShape(tracks)
    covariance_ellipses_pred = ComputeCovarianceEllipses(tracks, kPredictedColor)
    covariance_ellipses_corr = ComputeCovarianceEllipses(tracks, kPredictedColor)
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
                                               label='Corrrected Reproj')
    reproj_prefit_left_scatter = axd['left1'].scatter(reproj_left[0,:], 
                                                      reproj_left[1,:], 
                                                      edgecolors='red', facecolors='none', linewidths=2.0, s=80, 
                                                      label='Predicted Reproj')
    axd['left1'].legend(loc='lower right', fontsize=16)    
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
    axd['left3'].set_xlabel('Track Index',                   fontsize=label_font_size, weight='bold')
    axd['left3'].set_ylabel('Distance (pixels)',             fontsize=label_font_size, weight='bold')
    axd['left3'].legend(loc='upper right',                   fontsize=18)  
    axd['left3'].grid(True)
    axd['left3'].set_ylim(-2,15)
    # ----------------------------------------------------------------------------------------
    # Middle 1 PREDICT
    AddCovarianceEllipses(axd['middle1'], covariance_ellipses_pred.vertical_view)
    vert_pred_scatter = axd['middle1'].scatter(tracks.pts_3d_mean[0,:], 
                                               tracks.pts_3d_mean[2,:], 
                                               facecolors='crimson',   
                                               edgecolors='black',   
                                               linewidths=2.5,        
                                               s=kTrackScttrSize,
                                               zorder=2)
    axd['middle1'].set_xlim((image_extent_vertical[0], image_extent_vertical[1]))
    axd['middle1'].set_ylim((image_extent_vertical[2], image_extent_vertical[3]))
    axd['middle1'].set_xlabel('X Lateral Axis (m)', fontsize=label_font_size,   weight='bold')
    axd['middle1'].set_ylabel('Vertical View',      fontsize=title_font_size,   weight='bold')
    axd['middle1'].set_title('Predicted',           fontsize=title_font_size+5, weight='bold')
    axd['middle1'].grid(True, color='violet',       linestyle='--', linewidth=0.5)
    axd['middle1'].set_aspect("equal")
    # Middle 2 PREDICT
    AddCovarianceEllipses(axd['middle2'], covariance_ellipses_pred.lateral_view)
    lateral_pred_scatter = axd['middle2'].scatter(tracks.pts_3d_mean[2,:], 
                                                  tracks.pts_3d_mean[1,:], 
                                                  facecolors='crimson',   
                                                  edgecolors='black',   
                                                  linewidths=2.5,        
                                                  s=kTrackScttrSize,
                                                  zorder=2)                 
    axd['middle2'].set_xlim((image_extent_lateral[0], image_extent_lateral[1]))
    axd['middle2'].set_ylim((image_extent_lateral[2], image_extent_lateral[3]))
    axd['middle2'].set_xlabel('Z Longitudinal Axis (m)', fontsize=label_font_size, weight='bold')
    axd['middle2'].set_ylabel('Side View', fontsize=title_font_size, weight='bold')
    axd['middle2'].grid(True, color='violet', linestyle='--', linewidth=0.5)
    axd['middle2'].set_aspect("equal")
    # ----------------------------------------------------------------------------------------
    # Right 1 CORRECT
    AddCovarianceEllipses(axd['right1'], covariance_ellipses_corr.vertical_view)
    vert_corr_scatter = axd['right1'].scatter(tracks.pts_3d_mean[0,:], 
                                              tracks.pts_3d_mean[2,:], 
                                              facecolors='lime',   
                                              edgecolors='black',   
                                              linewidths=2.5,        
                                              s=kTrackScttrSize,
                                              zorder=2)
    axd['right1'].set_xlim((image_extent_vertical[0], image_extent_vertical[1]))
    axd['right1'].set_ylim((image_extent_vertical[2], image_extent_vertical[3]))
    axd['right1'].set_xlabel('X Lateral Axis (m)', fontsize=label_font_size, weight='bold')
    axd['right1'].set_ylabel('Z Longitudinal Axis (m)', fontsize=label_font_size, weight='bold')
    axd['right1'].set_title('Corrected', fontsize=title_font_size+5, weight='bold')
    axd['right1'].grid(True, color='violet', linestyle='--', linewidth=0.5)
    axd['right1'].set_aspect("equal")
    # Right 2 CORRECT
    AddCovarianceEllipses(axd['right2'], covariance_ellipses_corr.lateral_view)
    lateral_corr_scatter = axd['right2'].scatter(tracks.pts_3d_mean[2,:], 
                                                 tracks.pts_3d_mean[1,:], 
                                                 facecolors='lime',   
                                                 edgecolors='black',   
                                                 linewidths=2.5,        
                                                 s=kTrackScttrSize,
                                                 zorder=2)                 
    axd['right2'].set_xlim((image_extent_lateral[0], image_extent_lateral[1]))
    axd['right2'].set_ylim((image_extent_lateral[2], image_extent_lateral[3]))
    axd['right2'].set_xlabel('Z Longitudinal Axis (m)', fontsize=label_font_size, weight='bold')
    axd['right2'].set_ylabel('Y Vertical Axis (m)', fontsize=label_font_size, weight='bold')
    axd['right2'].grid(True, color='violet', linestyle='--', linewidth=0.5)
    axd['right2'].set_aspect("equal")
    #
    fig.text(
        0.98, 0.03,                                                    # x=98% right, y=2% from bottom of figure
        "90% Confidence Ellipses of Track Covariances",
        ha='right',                                                    # Align right edge of text to x=0.98
        va='bottom',                                                   # Align bottom edge of text to y=0.02
        color='black',                                                 # High contrast for your black background
        fontsize=40,
        style='italic'                                                 # Optional subtle caption styling
    )
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
    # Middle 2
    plot_handles.lateral_pred_scatter = lateral_pred_scatter
    # Right 1
    plot_handles.vert_corr_scatter = vert_corr_scatter               
    # Right 2
    plot_handles.lateral_corr_scatter = lateral_corr_scatter
    
    return plot_handles
    
def UpdatePredictedXSectionViews(plot_handles,
                                 tracks):
    """Update Prediction Tracks Cross-Sectional Views
    Args:
        plot_handles:
        tracks: 
    Returns:
    """
    covariance_ellipses = ComputeCovarianceEllipses(tracks, kPredictedColor)
    # axd['middle1']
    AddCovarianceEllipses(plot_handles.axd['middle1'], covariance_ellipses.vertical_view)
    plot_handles.vert_pred_scatter.set_offsets(np.vstack((tracks.pts_3d_mean[0,:], tracks.pts_3d_mean[2,:])).T)
    # axd['middle2']
    AddCovarianceEllipses(plot_handles.axd['middle2'], covariance_ellipses.lateral_view)
    plot_handles.lateral_pred_scatter.set_offsets(np.vstack((tracks.pts_3d_mean[2,:], tracks.pts_3d_mean[1,:])).T)

    return

def UpdateCorrectedXSectionViews(plot_handles,
                                 tracks):
    """Update Corrected Tracks Cross-Sectional Views
    Args:
        plot_handles:
        tracks: 
    Returns:
    """
    covariance_ellipses = ComputeCovarianceEllipses(tracks, kCorrectedColor)
    # axd['right1']
    AddCovarianceEllipses(plot_handles.axd['right1'], covariance_ellipses.vertical_view)
    plot_handles.vert_corr_scatter.set_offsets(np.vstack((tracks.pts_3d_mean[0,:], tracks.pts_3d_mean[2,:])).T)
    # axd['right2']
    AddCovarianceEllipses(plot_handles.axd['right2'], covariance_ellipses.lateral_view)
    plot_handles.lateral_corr_scatter.set_offsets(np.vstack((tracks.pts_3d_mean[2,:], tracks.pts_3d_mean[1,:])).T)

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
    assert pts_3d_mean_pred.shape == (3, num_points)
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
    #width, height = fig.canvas.get_width_height()
    img_rgba = np.asarray(fig.canvas.buffer_rgba())
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    # 4. Extract the EXACT physical pixel dimensions from the shape
    # NumPy shape order is (Height, Width, Channels)
    true_height, true_width, _ = img_bgr.shape
    writer = cv2.VideoWriter(file_name, 
                             fourcc, 
                             kfps, 
                             (true_width, true_height)) 
    
    return writer

def WriteToVideo(writer, fig):
    img_rgba = np.asarray(fig.canvas.buffer_rgba())
    # OpenCV requires BGR format, drop the Alpha channel and reverse RGB -> BGR
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    # Write the frame to the video file
    writer.write(img_bgr)
    return

