In this directory, I am experimenting with a setup of tracking the tip
of the cautery pen with the aid of aruco board.

## History:
* Started out with EstimatePostureWithArucoGridboard_realsense.py
*  EstimatePostureWithArucoGridboard_realsense_classTest.py: Then added
   aruco_board_tracker.py class, and created this class based program.
* dope_aruco_tracking.py: for live realsense based dope+aruco tracking
  which produces a combined frame with both aruco and dope tracking
  results.
* dope_aruco_tracking_videoMaker.py: Made a video recorder. The program
  creates 2 video outputs: input.avi and dope_aruco_annotated.avi. The
  first one contains the original frames captured using realsense. The
  second one saves the aruco+dope tracking results. Frame rate is messed
  up. But can be worked with.
* dope_aruco_tracking_videoReader.py: This will read input.avi and do
  analysis of tip tracking. Possibly this will be the final analysis
  code.  
  I am adding a .xls file creation part so that later on a different
  program I can analyze.
  
  
## Notes:
### Folders
"raw_videos" folder will contain raw captured input videos and the
dope_aruco_annotated videos. Later these input videos will be edited and
exported to "output" folder. "video_editing" folders will contain
OpenShot project fiels for each input videos.

### Translation matrix  
After calculation I have determined that the translation matrix that
will convert the {A} (orignial center of the cuboid) camera coordinate
to {B} the new coordinate centered at the center of the bottom plane is
\[0,-12.5,0]. Just have add this to the cuboid3d_points in
cuboid_pnp_solver to get the representations with respect to {B}.

### Final fix for coordinate transfer  
The fix was easy. In the main code just initializing the Cuboid3D
object with center location = \[0. -12.5, 0] converted coordinate
system.

### Video Editing
I inted to edit all input videos. I'll only eliminate frames where both
hand and board are not PRESENT. This will give me a lot of statistics
about the tracking and its accuracy. 