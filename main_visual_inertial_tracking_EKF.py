"""
        Stereo Visual Inertial Object Tracking with EKF

Usage: python3 main_visual_inertial_tracking_EKF.py
Requirements: Numpy, Matplotlib, scipy, cv2, skimage, sympy
"""

import SVIT_EKF.vitekf_core           as core
import SVIT_EKF.vitekf_utils_core     as utils
import SVIT_EKF.vitekf_utils_oxts     as utils_oxts
import SVIT_EKF.vitekf_utils_plotting as utils_plt
import matplotlib.pyplot              as plt
import copy

if __name__ == "__main__":
    data_path = '2011_09_26/2011_09_26_drive_0001_sync/'
    oxts_data = utils_oxts.GetOXTSDataFull(data_path)
    images    = utils.GetImagePair(data_path, index = 0)
    tracks    = core.InitializeTracks(images)
    ##
    plt_hndls = utils_plt.InitializeVisualization(images, tracks)
    plt.draw()
    plt.pause(3.0)
    writer = utils_plt.InitializeVideoWriter(plt_hndls.fig, 'SVIT_IEKF_0001.mp4')
    for _ in range(0,10):
        utils_plt.WriteToVideo(writer, plt_hndls.fig)
    ##
    for index in range(1, oxts_data.num_samples):
        # ------------------ I. Prediction Step ------------------  
        tracks = core.ForwardPropagate3DPoints(index, oxts_data, tracks)
        # Update Plots
        utils_plt.UpdatePredictedXSectionViews(plt_hndls, tracks)
        # ------------------ II. Track 2d-Features in Next Image\Measurement ------------------ 
        images_next = utils.GetImagePair(data_path, index)
        # Update Plots
        tracks =  core.TrackFeaturesAndReplaceLostPoints(images, # From Last Cycle
                                                         tracks, # From Last Cycle
                                                         images_next)
        images = copy.deepcopy(images_next) # Update for Next Cycle
        pts_3d_mean_pred = tracks.pts_3d_mean.copy()
        # ------------------- III. Measurement Correction Step -------------------
        tracks = core.KFMeasurementCorrectionStep(tracks)
        # Update Plots
        utils_plt.UpdateImageAndReprojectionError(plt_hndls, 
                                                  images_next, 
                                                  tracks,
                                                  pts_3d_mean_pred)
        utils_plt.UpdateCorrectedXSectionViews(plt_hndls, tracks)
        plt.draw()
        plt.pause(0.1)
        utils_plt.WriteToVideo(writer, plt_hndls.fig)

    writer.release()
    plt.show()
