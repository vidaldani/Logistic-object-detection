#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import pandas as pd
import math
from geometry_msgs.msg import Pose, PoseStamped
from gazebo_msgs.msg import ModelStates
from turtlebot3_recognition.msg import BoundingBox3D
import tf_transformations #as tf
import tf2_ros
import tf2_geometry_msgs
import os

import geometry_msgs.msg
from tf2_geometry_msgs import do_transform_pose
from datetime import datetime




class DataCollectorNode(Node):
    def __init__(self):
        super().__init__('data_collector_node')

        # Initialize variables
        self.data = []
        self.detection_count = 0
        self.max_detections = 100

        # Subscribers
        self.bounding_box_subscriber = self.create_subscription(
            BoundingBox3D,
            '/bounding_boxes_3d',
            self.bounding_box_callback,
            10
        )

        self.gazebo_subscriber = self.create_subscription(
            ModelStates,
            '/model_states',
            self.gazebo_callback,
            10
        )

        # Initialize variables to store ground truth poses
        self.ground_truth_poses = {}
        self.ground_truth_dimensions = self.load_model_dimensions()

        # Mapping from detection class names to Gazebo model names
        self.class_to_model_name = {
            #'forklift': 'forklift_loaded_stillage', #'toyota_forklift',
            'forklift': 'toyota_forklift', #'forklift_loaded_stillage',
            'pallet': 'wooden pallet',
            'small_load_carrier': 'KLT',
            'stillage': 'stillage_box',
            'pallet_truck': 'aws_robomaker_warehouse_PalletJackB_01'
        }

        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera frame
        # self.camera_frame = 'camera_rgb_frame'  #'realsense_rgb_frame' Or 'camera_rgb_optical_frame'  Camera frame
        # self.world_frame = 'base_link'  # World frame is odom, can be changed

        self.get_logger().info("DataCollectorNode initialized.")

    def gazebo_callback(self, msg):
        # Store ground truth poses
        for name, pose in zip(msg.name, msg.pose):
            self.ground_truth_poses[name] = pose

    def bounding_box_callback(self, msg):
        if self.detection_count >= self.max_detections:
            return

        # Map object_name to model_name
        model_name = self.class_to_model_name.get(msg.object_name)
        if model_name is None:
            self.get_logger().warn(f'No model mapping for object name: {msg.object_name}')
            return

        # Get ground truth pose
        gt_pose = self.ground_truth_poses.get(model_name)
        if gt_pose is None:
            self.get_logger().warn(f'Ground truth pose for {model_name} not found')
            return

        # Get ground truth dimensions
        gt_dimensions = self.ground_truth_dimensions.get(model_name)
        if gt_dimensions is None:
            self.get_logger().warn(f'Ground truth dimensions for {model_name} not found')
            return

        # Extract x, y, z, yaw from detected pose
        # detected_pose = msg.center  # Pose of the detected object
        # x = detected_pose.position.x
        # y = detected_pose.position.y
        # z = detected_pose.position.z
        # yaw = self.quaternion_to_yaw(detected_pose.orientation)

        # Transform the detected pose from 'realsense_depth_frame' to 'world' frame
        detected_pose_stamped = PoseStamped()
        detected_pose_stamped.header.frame_id = 'realsense_depth_frame'
        detected_pose_stamped.header.stamp = self.get_clock().now().to_msg()  # Add timestamp
        
        # Set the detected pose from msg.center (which is a Pose type)
        detected_pose_stamped.pose = msg.center

        # Print the frame id to check if it's correct
        self.get_logger().info(f"Bounding box detected in frame: {msg.frame_id}")
        
        try:
            # Transform to the 'world' frame
            transform = self.tf_buffer.lookup_transform(
                'odom',
                # msg.frame_id,  # Use the frame provided in BoundingBox3D message
                'realsense_depth_frame',
                rclpy.time.Time(),  # Use time 0 to get the latest available transform
                timeout=rclpy.duration.Duration(seconds=1.0)  # 1-second timeout 
                # rclpy.time.Time()
            )
            
            transformed_pose_stamped = tf2_geometry_msgs.do_transform_pose(detected_pose_stamped.pose, transform)
            # self.get_logger().info(f"transformed_pose_stamped: {transformed_pose_stamped}")

            # Extract x, y, z, yaw from the transformed pose
            x = transformed_pose_stamped.position.x
            y = transformed_pose_stamped.position.y
            z = transformed_pose_stamped.position.z
            yaw = self.quaternion_to_yaw(transformed_pose_stamped.orientation)
            
        except Exception as e:
            self.get_logger().error(f"Failed to transform detected pose: {e}")
            return
        
        # Center pose (x, y)
        # center_pose_x = x
        # center_pose_y = y

        # Get camera pose relative to the world frame
        # try:
        #     now = self.get_clock().now().to_msg()  # Correct time handling
        #     #now = rclpy.time.Time()
        #     trans = self.tf_buffer.lookup_transform(
        #         self.world_frame,
        #         self.camera_frame,
        #         #rclpy.time.Time(),  # Or use now if needed
        #         now,
        #         timeout=rclpy.duration.Duration(seconds=2.0)
        #     )
        #     camera_position = trans.transform.translation
        #     camera_x = camera_position.x
        #     camera_y = camera_position.y
        #     camera_z = camera_position.z
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException): #as ex:
        #     self.get_logger().warn('TF lookup failed for camera pose:') #{ex}')
        #     return

        # Compute Euclidean distance
        # distance = self.calculate_distance((x, y, z), (camera_x, camera_y, camera_z))

        # Ground truth pose
        gt_x = gt_pose.position.x
        gt_y = gt_pose.position.y
        gt_z = gt_pose.position.z
        # Adjust ground truth z to match the center of the 3D box
        gt_adj_z = gt_z + (gt_dimensions['height'] / 2)
        gt_yaw = self.quaternion_to_yaw(gt_pose.orientation)

        # Compute Euclidean distance between detected object and ground truth object
        distance = self.calculate_distance((x, y, z), (gt_x, gt_y, gt_adj_z))

        # Detected bounding box dimensions
        det_length = msg.size.x
        det_width = msg.size.z
        det_height = msg.size.y

        # Compute IoU
        det_box = {
            'x': x,
            'y': y,
            'z': z,
            'length': det_length,
            'width': det_width,
            'height': det_height
        }
        gt_box = {
            'x': gt_x,
            'y': gt_y,
            'z': gt_adj_z,
            'length': gt_dimensions['length'],
            'width': gt_dimensions['width'],
            'height': gt_dimensions['height']
        }
        iou = self.compute_iou(det_box, gt_box)

        # Append data
        self.data.append({
            'x': x,
            'y': y,
            'z': z,
            'yaw': yaw,
            # 'center_pose': (center_pose_x, center_pose_y),
            'distance_to_ground_truth': distance,
            'gt_x': gt_x,
            'gt_y': gt_y,
            'gt_z': gt_adj_z,
            'gt_yaw': gt_yaw,
            'length': det_length,
            'width': det_width,
            'height': det_height,
            'gt_length': gt_dimensions['length'],
            'gt_width': gt_dimensions['width'],
            'gt_height': gt_dimensions['height'],
            'iou': iou
        })

        self.detection_count += 1
        self.get_logger().info(f'Detection {self.detection_count} recorded with IoU: {iou}')

        if self.detection_count == self.max_detections:
            self.save_to_excel()
            self.get_logger().info('Collected 100 detections, shutting down.')
            rclpy.shutdown()

    def calculate_distance(self, position1, position2):
        dx = position1[0] - position2[0]
        dy = position1[1] - position2[1]
        dz = position1[2] - position2[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)

    def quaternion_to_yaw(self, orientation):
        euler = tf_transformations.euler_from_quaternion([
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        ])
        yaw = euler[2]
        return yaw

    def compute_iou(self, det_box, gt_box):
        # Calculate min and max coordinates for detected box
        det_min = {
            'x': det_box['x'] - det_box['length'] / 2,
            'y': det_box['y'] - det_box['width'] / 2,
            'z': det_box['z'] - det_box['height'] / 2
        }
        det_max = {
            'x': det_box['x'] + det_box['length'] / 2,
            'y': det_box['y'] + det_box['width'] / 2,
            'z': det_box['z'] + det_box['height'] / 2
        }

        # Calculate min and max coordinates for ground truth box
        gt_min = {
            'x': gt_box['x'] - gt_box['length'] / 2,
            'y': gt_box['y'] - gt_box['width'] / 2,
            'z': gt_box['z'] - gt_box['height'] / 2
        }
        gt_max = {
            'x': gt_box['x'] + gt_box['length'] / 2,
            'y': gt_box['y'] + gt_box['width'] / 2,
            'z': gt_box['z'] + gt_box['height'] / 2
        }

        # self.get_logger().info(f'Detected box min: {det_min}, max: {det_max}')
        # self.get_logger().info(f'Ground truth box min: {gt_min}, max: {gt_max}')

        # Calculate overlap in each dimension
        overlap_min = {
            'x': max(det_min['x'], gt_min['x']),
            'y': max(det_min['y'], gt_min['y']),
            'z': max(det_min['z'], gt_min['z'])
        }
        overlap_max = {
            'x': min(det_max['x'], gt_max['x']),
            'y': min(det_max['y'], gt_max['y']),
            'z': min(det_max['z'], gt_max['z'])
        }

        # Check for overlap
        overlap_dims = {
            'x': max(0.0, overlap_max['x'] - overlap_min['x']),
            'y': max(0.0, overlap_max['y'] - overlap_min['y']),
            'z': max(0.0, overlap_max['z'] - overlap_min['z'])
        }

        # Compute volumes
        intersection_volume = overlap_dims['x'] * overlap_dims['y'] * overlap_dims['z']

        det_volume = det_box['length'] * det_box['width'] * det_box['height']
        gt_volume = gt_box['length'] * gt_box['width'] * gt_box['height']

        union_volume = det_volume + gt_volume - intersection_volume

        if union_volume == 0.0:
            return 0.0  # Avoid division by zero

        iou = intersection_volume / union_volume
        return iou

    def load_model_dimensions(self):
        # Manually define the dimensions
        model_dimensions = {
            'toyota_forklift': {'length': 2.44, 'width': 1.15, 'height': 2.1},
            'wooden pallet': {'length': 1.2475, 'width': 0.8325, 'height': 0.146},
            'KLT': {'length': 0.594, 'width': 0.395, 'height': 0.278},
            'stillage_box': {'length': 1.28, 'width': 0.855, 'height': 1.0},
            'aws_robomaker_warehouse_PalletJackB_01': {'length': 0.570, 'width': 0.450, 'height': 0.966}
        }
        return model_dimensions

    def save_to_excel(self):
        # Convert data to DataFrame
        df = pd.DataFrame(self.data)
        # Generate the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Format to display only 3 decimal places in Excel (without affecting the internal precision)
        # pd.options.display.float_format = '{:.3f}'.format

        # Save to Excel file
        file_name = f'detection_data_{timestamp}_SAM2b+_Forklift_no_or.xlsx'
        # df.to_excel(file_name, index=False)

        with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)

            # Get the xlsxwriter objects from the DataFrame writer object
            workbook  = writer.book
            worksheet = writer.sheets['Sheet1']

            # Create a number format for 3 decimal places
            number_format = workbook.add_format({'num_format': '0.000', 'align': 'center'})
            text_format = workbook.add_format({'align': 'center'})  # For non-numeric columns

            # Apply the format to all the data columns (1-based index, adjust if needed)
            # worksheet.set_column(0, len(df.columns) - 1, None, number_format)

            # Set a minimum width for short columns and cap the width for long data
            min_width = 6    # Minimum width for very short column names like "x", "y", etc.
            max_width = 24   # Maximum width to avoid excessively wide columns for numbers

            # Loop through each column and adjust the column width
            for i, column in enumerate(df.columns):
                # Find the maximum length of the column title and the data
                if pd.api.types.is_numeric_dtype(df[column]):
                    # If it's a numeric column, set a smaller max width
                    column_len = max(len(column), 8)  # Minimum width for number columns
                    worksheet.set_column(i, i, column_len, number_format)  # Apply number format
                else:
                    # For non-numeric columns, calculate based on content length
                    column_len = max(df[column].astype(str).map(len).max(), len(column))
                    worksheet.set_column(i, i, column_len, text_format)  # Apply text format
                    
                # Cap the column length between min_width and max_width
                column_len = max(min_width, min(column_len, max_width))

                # Set the column width
                # worksheet.set_column(i, i, column_len, number_format)

        self.get_logger().info(f'Data saved to {file_name}')

def main(args=None):
    rclpy.init(args=args)
    node = DataCollectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
