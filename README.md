This project is recommended to run on Ubuntu 22.04, Gazebo Sim 8.9.0 (Harmonic), and ROS 2 Humble. Please download the PX4 source code in advance.
When using PX4 v1.14+ on Ubuntu 22.04, the Gazebo instance automatically downloaded by PX4 may not communicate with ROS 2 via the bridge package, so a manual bridge is required.
1. Set environment variables
```bash
export GZ_VERSION=harmonic
```
2. Create a workspace
```bash
mkdir -p ~/ws/src
cd ~/ws/src
```
3. Clone the source code
```bash
git clone https://github.com/gazebosim/ros_gz.git -b humble
```
4. Install dependencies
```bash
cd ~/ws
rosdep install -r --from-paths src -i -y --rosdistro humble
```
5. Build the workspace
```bash
source /opt/ros/humble/setup.bash
cd ~/ws
colcon build
```
Then start the PX4 simulation and bridge:
cd PX4-Autopilot/
make px4_sitl gz_x500_mono_cam
ros2 run ros_gz_bridge parameter_bridge /world/default/model/x500_mono_cam_0/link/camera_link/sensor/imager/image@sensor_msgs/msg/Image@gz.msgs.Image
After that, you can view the corresponding image topic in RViz2.

Now run the project: first start the PX4 UAV simulation model with a camera, bridge the Gazebo camera topic to ROS 2 (as shown in the manual bridging tutorial above), and then run:
ros2 run gazebo_yolo_detection gazebo_yolo_detector.py
Note: only the source code of gazebo_yolo_detector.py is provided. You need to create it as a ROS 2 node (Node) yourself before it can be run via ros2 run.
Then you can observe the UAV camera feed in RViz2. You can also add some objects in the world to verify the YOLO detection results.
After connecting the D435, run python3 allhand.py, and then run python3 gaze_visualization_node.py in another terminal. You can then control the UAV from the first-person view in RViz2 by drawing trajectories with hand gestures.
Note: the topic names used in gazebo_yolo_detector.py, allhand.py, and gaze_visualization_node.py need to be modified according to your actual topic names.
