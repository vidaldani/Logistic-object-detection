#!/usr/bin/env python3

# Copyright 2024 fml
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Daniel Vidal daniel.vidal@tum.de */

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Quaternion, Pose, Vector3
from turtlebot3_recognition.msg import Yolov8Inference, BoundingBox3D

import cv2
import numpy as np
from cv_bridge import CvBridge
import tf_transformations as tf

class YoloDepthSegmentationNode(Node):

    def __init__(self):
        super().__init__('yolo_depth_segmentation_node')
        
        

        # Initialize a CvBridge to convert between ROS and OpenCV images
        self.bridge = CvBridge()

        # Subscribe to the depth image topic
        self.depth_subscriber = self.create_subscription(
            Image,
            '/depth_camera/depth/image_raw',
            self.depth_image_callback,
            10
        )
        
        # Subscribe to the camera info topic to get camera parameters
        self.camera_info_subscriber = self.create_subscription(
            CameraInfo,
            '/depth_camera/depth/camera_info',
            self.camera_info_callback,
            10
        )

        # Subscribe to the Yolov8 inference results topic
        self.inference_subscriber = self.create_subscription(
            Yolov8Inference,
            '/Yolov8_Inference',
            self.inference_callback,
            10
        )
        
        # Publisher for the 3D bounding box
        self.publisher_ = self.create_publisher(BoundingBox3D, '/bounding_boxes_3d', 10)

        self.frame_id = None
        self.camera_info = None
        self.depth_image = None
        self.bounding_boxes = []
        
    def camera_info_callback(self, msg):
        # Store the camera info message for later use
        self.camera_info = msg
    

    def depth_image_callback(self, msg):
        # Convert the ROS Image message to an OpenCV image
        self.depth_image = self.bridge.imgmsg_to_cv2(img_msg=msg, desired_encoding='passthrough')
        self.frame_id = msg.header.frame_id

    
    def inference_callback(self, msg):
        # Store the bounding boxes from the Yolov8 Inference message
        self.bounding_boxes = msg.yolov8_inference
        
        # Filter and store bounding boxes only if the class_name is "forklift"
        # self.bounding_boxes = [box for box in msg.yolov8_inference if box.class_name == 'forklift']
        # Check if we have bounding boxes to crop the depth image
        if self.bounding_boxes and self.depth_image is not None:
            for box in self.bounding_boxes:
                top, left, bottom, right = round(box.left/2), round(box.top/2), round(box.right/2),  round(box.bottom/2)
                cropped_depth_image = self.depth_image[top:bottom, left:right]
                class_name = box.class_name
                confidence = box.conf
                
                # Visualize depth image with BB for debugging
                # cv2.rectangle(self.depth_image, (left, top), (right, bottom), (0, 255, 0), 2)
                # self.visualize_depth_image(self.depth_image, 'Full Depth Image')
                
                # Apply depth filtering
                filtered_image = self.apply_depth_filter(cropped_depth_image)
                
                # Display the filtered depth image for debugging
                # self.visualize_depth_image(filtered_image, 'Filtered Depth Image')
                
                # Transform the filtered depth mask into a 3D bounding box and publish as a marker
                self.publish_3d_marker(filtered_image, top, left, class_name, confidence)

        
    def apply_depth_filter(self, depth_image):
        # Flatten the depth image to analyze the depth values
        depth_values = depth_image.flatten()
        depth_values = depth_values[depth_values > 0]  # Exclude zero (no data) values

        if len(depth_values) == 0:
            return depth_image

        # Calculate the histogram of depth values
        hist, bin_edges = np.histogram(depth_values, bins=100)

        # Find the bin with the maximum count (the most common depth)
        max_bin_index = np.argmax(hist)
        dominant_depth = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

        # Define a range around the dominant depth to keep (e.g., +- 10% of the dominant depth)
        depth_range = 0.1 * dominant_depth
        lower_bound = dominant_depth - depth_range
        upper_bound = dominant_depth + depth_range

        # Create a mask to filter out depth values outside the range
        mask = (depth_image >= lower_bound) & (depth_image <= upper_bound)

        # Apply the mask to the depth image
        filtered_image = np.where(mask, depth_image, 0)

        return filtered_image


    def calculate_orientation_from_bbox(self, min_x, z_min_x, x_min_z, min_z):
        # Calculate the angle (theta) in the XZ plane
        # This is the angle of the line formed between (min_x, z_min_x) and (x_min_z, min_z)
        theta = np.arctan2(min_z - z_min_x, x_min_z - min_x)
        #print(min_x, z_min_x, x_min_z, min_z)
        #print(theta)

        # Convert the angle to a quaternion
        # Since we're working in the XZ plane, we only need to apply a rotation around the Y-axis
        quaternion = tf.quaternion_from_euler(0, theta, 0)

        return quaternion
    

    def publish_3d_marker(self, filtered_image, top, left, class_name, conf):
        if filtered_image is None or self.camera_info is None:
            return 
        
        # Intrinsic parameters
        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # Get the non-zero depth values and their coordinates
        depth_values = filtered_image[filtered_image > 0]
        coords = np.column_stack(np.where(filtered_image > 0))

        if len(depth_values) == 0:
            return

        # Convert pixel coordinates to 3D world coordinates
        z = depth_values # Assuming depth is in millimeters
        x = (coords[:, 1] + left - cx) * z / fx
        y = (coords[:, 0] + top - cy) * z / fy
        
        # Calculate the 3D bounding box in the world coordinates
        min_x, min_y, min_z = float(np.min(x)), float(np.min(y)), float(np.min(z))
        max_x, max_y, max_z = float(np.max(x)), float(np.max(y)), float(np.max(z))
        
        # Get the index of the minimum X value
        min_x_index = np.argmin(x)
        z_min_x = z[min_x_index]  # Corresponding Z value for min_x
        
        # Get the index of the minimum Z value
        min_z_index = np.argmin(z)
        x_min_z = x[min_z_index]  # Corresponding X value for min_z
        
        # Set the marker's position (center of the bounding box)
        central_x = (min_x + max_x) / 2.0
        central_y = (min_y + max_y) / 2.0
        central_z = (min_z + max_z) / 2.0
        
        # Create the Pose object for transformation
        pose = Pose()
        pose.position.x = central_x
        pose.position.y = central_y
        pose.position.z = central_z
        
        quaternion =  self.calculate_orientation_from_bbox(min_x, z_min_x, x_min_z, min_z)
        pose.orientation.x = quaternion[0]
        pose.orientation.y = quaternion[1]
        pose.orientation.z = quaternion[2]
        pose.orientation.w = quaternion[3]
        
        # Create an instance of the BoundingBox3D message
        bounding_box_msg = BoundingBox3D()
        
        # Populate the message fields
        bounding_box_msg.object_name = class_name
        bounding_box_msg.conf = conf  # Example confidence value
        bounding_box_msg.center = pose
        
        # Populate the size (width, height, depth)
        size = Vector3()
        size.x = float(max_x - min_x)  # width
        size.y = float(max_y - min_y)  # height
        size.z = float(max_z - min_z)  # depth
        bounding_box_msg.size = size
        
        # Set the frame_id
        bounding_box_msg.frame_id = self.frame_id # Example frame ID
        # Publish the message
        self.publisher_.publish(bounding_box_msg)
            

    def visualize_depth_image(self, depth_image, window_name):
        # Normalize the depth image to the range [0, 255] for visualization
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Optionally, apply a colormap for better visualization (e.g., cv2.COLORMAP_JET)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Display the depth image using OpenCV
        cv2.imshow(window_name, depth_colored)
        cv2.waitKey(1)  # Refresh the image window    

def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthSegmentationNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()