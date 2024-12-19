# Logistic-object-detection
This project contains a Gazebo simulation with all the [LOCO](https://github.com/tum-fml/loco?tab=readme-ov-file) dataset and the chance to perform object detection using YOLOv8 n with this data.

## Logistic simulation

![Screenshot from 2024-08-06 15-18-29](https://github.com/user-attachments/assets/b46cba82-d582-4351-8cc5-56ba18048712)

1.  Clone repository
   ```console
   cd ros2_ws/src
   git clone https://github.com/vidaldani/Logistic-object-detection.git
   ```
2.  Compile workspace
   ```console
   cd ..
   colcon build
   source install/setup.bash 
   ```
3.  Export Turtlebot model
   ```console
   gedit ~/.bashrc
   # copy this instruction at the end: export TURTLEBOT3_MODEL=waffle
   source ~/.bashrc
   ```
4.  Copy the models inside /turtlebot3_gazebo/models/logistic_objects to ~/.gazebo/models folder
   ```console
   cp -r ~/ros2_ws/src/Logistic-object-detection/src/turtlebot3_gazebo/models/logistic_objects/* ~/.gazebo/models/
   ```
6.  Launch logistic simulation:
   ```console
   ros2 launch turtlebot3_gazebo versuchshalle.launch.py
   ```
## Launch object detection

![Screencast-from-08-06-2024-03_02_37-PM](https://github.com/user-attachments/assets/5269630e-3364-47ef-b55b-98c26c450179)

1. Install ultralytics (if it is already installed jump to step 2)
   ```console
   pip3 install ultralytics==8.3.40
   ```
2. On a different terminal launch logistic object detection
   ```console
   cd ros2_ws/
   source install/setup.bash 
   ros2 launch turtlebot3_recognition launch_yolov8.launch.py
   ```
3. Open rviz to visualize the inference results. The configuration file can be found in the folder /turtlebot3_recognition/rviz
   ```console
   rviz2 -d $(pwd)/src/turtlebot3_ws/src/turtlebot3_recognition/rviz/rviz.rviz
   ```
5. In a new terminal launch the keyboard teleoperation to move the robot around and perform object detection
   ```console
   ros2 run teleop_twist_keyboard teleop_twist_keyboard
   ```
## 3D object detection
![Screencast-from-08-21-2024-04_42_29-PM](https://github.com/user-attachments/assets/1135a029-271b-4e1a-8baa-f2ccb37c6277)

1. Launch 3D object detection
   ```console
   ros2 launch turtlebot3_recognition launch_3d_node.launch.py
   ```



