#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Quaternion, Pose, Vector3
from turtlebot3_recognition.msg import Yolov8Inference, BoundingBox3D

from segment_anything import sam_model_registry, SamPredictor

import cv2
import numpy as np
from cv_bridge import CvBridge
import tf_transformations as tf
from sklearn.decomposition import PCA  # Import PCA for orientation calculation

import time
from std_msgs.msg import Float32

class YoloSAMFilterSegmentationNode(Node):

    def __init__(self):
        super().__init__('yolo_sam_filter_segmentation_node')
        
        # Initialize SAM Model for segmentation
        self.sam_model = sam_model_registry["vit_b"](checkpoint="/home/tumwfml-ubunt6/SAM_models/sam_vit_b_01ec64.pth") # Change directory if needed
        # self.sam_model = sam_model_registry["vit_b"](checkpoint="/home/user/models/sam_vit_b_01ec64.pth") # Change directory if needed
        # self.sam_model = sam_model_registry["vit_l"](checkpoint="/home/user/models/sam_vit_l_0b3195.pth") # Change directory if needed
        # self.sam_model = sam_model_registry["vit_h"](checkpoint="/home/user/models/sam_vit_h_4b8939.pth") # Change directory if needed
        
        self.predictor = SamPredictor(self.sam_model)        

        # Initialize a CvBridge to convert between ROS and OpenCV images
        self.bridge = CvBridge()

        # Publishers for inference times
        self.sam_segmentation_time_publisher = self.create_publisher(Float32, "/sam_segmentation_time", 10)
        self.depth_filtering_time_publisher = self.create_publisher(Float32, "/depth_filtering_time", 10)
        self.orientation_time_publisher = self.create_publisher(Float32, "/orientation_time", 10)

        # Subscribe to the depth image topic
        self.depth_subscriber = self.create_subscription(
            Image,
            '/depth_camera/depth/image_raw',
            self.depth_image_callback,
            10
        )

        # Subscribe to the RGB camera image topic (needed for SAM)
        self.rgb_subscriber = self.create_subscription(
            Image,
            '/rgb_camera/rgb_camera/image_raw',
            self.rgb_image_callback,
            10
        )

        # Subscribe to the rgb camera info topic to get camera parameters
        self.rgb_camera_info_subscriber = self.create_subscription(
            CameraInfo,
            '/rgb_camera/rgb_camera/camera_info',
            self.rgb_camera_info_callback,
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

        
        # Publisher for the segmented RGB image with mask
        self.segmented_image_publisher = self.create_publisher(Image, '/segmented_rgb_image', 10)


        self.filtered_image_publisher = self.create_publisher(Image, '/filtered_depth_image', 10)
        self.mask_on_depth_publisher = self.create_publisher(Image, '/mask_on_depth_image', 10)

        self._filtered_image_publisher = self.create_publisher(Image, '/vidal_filtered_depth_image', 10)
        
        self.cropped_image_to_depth_publisher = self.create_publisher(Image, '/cropped_image_to_depth', 10)


        self.frame_id = None
        self.camera_info = None
        self.rgb_camera_info = None
        self.depth_image = None
        self.rgb_image = None  # Store the RGB image for SAM processing
        self.bounding_boxes = []

        # Add logging to track callbacks and operations
        self.get_logger().info("YoloSAMFilterSegmentationNode initialized.")
        
    def camera_info_callback(self, msg):
        # Store the camera info message for later use
        self.camera_info = msg
    
    def rgb_camera_info_callback(self, msg):
        # Store the rgb  camera info message for later use
        self.rgb_camera_info = msg

    def depth_image_callback(self, msg):
        # Convert the ROS Image message to an OpenCV image
        self.depth_image = self.bridge.imgmsg_to_cv2(img_msg=msg, desired_encoding='passthrough')
        self.frame_id = msg.header.frame_id
        self.get_logger().info(f"Received depth image of shape {self.depth_image.shape}")

    
    def rgb_image_callback(self, msg):
        # Convert the ROS RGB Image message to an OpenCV image
        self.rgb_image = self.bridge.imgmsg_to_cv2(img_msg=msg, desired_encoding='bgr8')
        self.get_logger().info(f"Received RGB image of shape {self.rgb_image.shape}")

    
    def inference_callback(self, msg):
        # Store the bounding boxes from the Yolov8 Inference message
        self.get_logger().info("Received YOLOv8 inference result.")    
        # Check if the depth_image and rgb_image are initialized and contain data
        if self.depth_image is None or self.rgb_image is None:
            self.get_logger().info("No depth or RGB image available yet.")
            return

        if self.depth_image.size == 0 or self.rgb_image.size == 0:
            self.get_logger().info("Depth or RGB image is empty.")
            return

        
        self.bounding_boxes = msg.yolov8_inference
        self.get_logger().info("Processing YOLOv8 bounding boxes...")

        # Filter and store bounding boxes only if the class_name is "forklift"
        # self.bounding_boxes = [box for box in msg.yolov8_inference if box.class_name == 'forklift']
        # Check if we have bounding boxes to crop the depth image
        if self.bounding_boxes and self.depth_image is not None and self.rgb_image is not None:
            start_time_sam_setting_img = time.time() 
            self.predictor.set_image(self.rgb_image)
            sam_setting_img_time = time.time() - start_time_sam_setting_img
            self.get_logger().info(f"SAM Setting Image Time: {sam_setting_img_time:.4f} seconds")
            for box in self.bounding_boxes:
                # top, left, bottom, right = round(box.left/2), round(box.top/2), round(box.right/2),  round(box.bottom/2)
                top, left, bottom, right = round(box.left), round(box.top), round(box.right),  round(box.bottom)
                _top, _left, _bottom, _right = round(box.left/2), round(box.top/2), round(box.right/2), round(box.bottom/2)
                
                cropped_depth_image = self.depth_image[_top:_bottom, _left:_right]
                
                depth_image = self.depth_image
                class_name = box.class_name
                confidence = box.conf
                
                # Visualize depth image with BB for debugging
                # cv2.rectangle(self.depth_image, (left, top), (right, bottom), (0, 255, 0), 2)
                # self.visualize_depth_image(self.depth_image, 'Full Depth Image')
                
                
                # Use SAM for segmentation on the RGB image
                height_depth_image, width_depth_image = cropped_depth_image.shape
                bounded_box_image = self.rgb_image[top:bottom, left:right]
                # cropped_image_to_depth_image = cv2.resize(bounded_box_image, (width_depth_image, height_depth_image))
                input_box = np.array([left, top, right, bottom])  # Box defined in full-resolution RGB image

                # Start the SAM segmentation timer
                start_time_sam = time.time()
                
                # Generate the mask using SAM based on the bounding box
                masks, _, _ = self.predictor.predict(box=input_box)

                # End the SAM segmentation timer and publish time
                sam_segmentation_time = time.time() - start_time_sam + sam_setting_img_time
                self.get_logger().info(f"SAM Segmentation Time: {sam_segmentation_time:.4f} seconds")
                self.sam_segmentation_time_publisher.publish(Float32(data=sam_segmentation_time))

                segmentation_mask = masks[0][top:bottom, left:right]  # Assuming the first mask is the best
                
                print("segmentation mask shape is:", segmentation_mask.shape)
                # Now we need to resize 
                resized_segmentation_mask = self.resize_segmentation_mask(segmentation_mask, height_depth_image, width_depth_image)
                print("segmentation mask is : ", segmentation_mask)
                print("segmentation shape is :", segmentation_mask.shape)
                print("resized segmentation mask shape is :", resized_segmentation_mask.shape)
                # print("cropped_image_to_depth shape is ", cropped_image_to_depth_image.shape)
                print("cropped_depth_image shape is ", cropped_depth_image.shape)

                # self.visualize_segmented_image(self.rgb_image, segmentation_mask)
                
                # Original method (depth filtering)
                # Apply depth filtering
                _filtered_image = self.apply_depth_filter(cropped_depth_image)
                # Publish _filtered_image for comparison
                self.publish_filtered_image(_filtered_image, "/vidal_filtered_depth_image")
 

                # Resize the mask to the size of the depth image
                # height, width = depth_image.shape
                # _resized_mask = cv2.resize(segmentation_mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

                # Visualize the resized mask on top of the depth image
                # self.visualize_mask_on_depth(_resized_mask, depth_image)


                # Apply the segmentation mask to the depth image
                filtered_image = self.apply_sam_mask_to_depth(resized_segmentation_mask, cropped_depth_image, _top, _left, _bottom, _right)
                
                # Normalize the cropped filtered image for visualization
                # normalized_filtered_image = cv2.normalize(cropped_filtered_image, None, 0, 255, cv2.NORM_MINMAX) if cropped_filtered_image.max() > 0 else cropped_filtered_image
        
                
                # Start the depth filtering timer
                start_time_depth = time.time()

                histogram_filtered_image = self.apply_depth_filter(filtered_image)
                
                # Publish filtered_image for comparison
                # self.publish_filtered_image(filtered_image, "/filtered_depth_image")
                self.publish_filtered_image(histogram_filtered_image, "/filtered_depth_image")

                # End the depth filtering timer and publish time
                depth_filtering_time = time.time() - start_time_depth
                self.get_logger().info(f"Depth Filtering Time: {depth_filtering_time:.4f} seconds")
                self.depth_filtering_time_publisher.publish(Float32(data=depth_filtering_time))

                # Publish filtered_image for comparison
                self.publish_filtered_image(bounded_box_image, "/cropped_image_to_depth")

                
                # Display the filtered depth image for debugging
                # self.visualize_depth_image(filtered_image, 'Filtered Depth Image')
                
                # Transform the filtered depth mask into a 3D bounding box and publish as a marker
                # self.publish_3d_marker(cropped_filtered_image, _top, _left, class_name, confidence)
                self.publish_3d_marker(histogram_filtered_image, _top, _left, class_name, confidence)
                # self.publish_3d_marker(filtered_image, _top, _left, class_name, confidence)
                

    def resize_segmentation_mask(self, matrix, target_n2, target_m2):
        n, m = matrix.shape
        target_n, target_m = target_n2 * 2, target_m2 * 2

        # Adjust rows
        if n > target_n:
            matrix = matrix[:target_n, :]  # Truncate rows
        elif n < target_n:
            padding_rows = target_n - n
            matrix = np.pad(matrix, ((0, padding_rows), (0, 0)), 'constant', constant_values=False)

        # Adjust columns
        if m > target_m:
            matrix = matrix[:, :target_m]  # Truncate columns
        elif m < target_m:
            padding_cols = target_m - m
            matrix = np.pad(matrix, ((0, 0), (0, padding_cols)), 'constant', constant_values=False)

        # Reshape and take the maximum of each 2x2 block
        return matrix.reshape(target_n2, 2, target_m2, 2).min(axis=(1, 3))



    def apply_sam_mask_to_depth(self, mask, depth_image, _top, _left, _bottom, _right):
        """
        This function applies the SAM-generated mask (from the RGB image) to the depth image.
        The depth image is filtered to only include points within the segmented mask.
        """

        # Apply the mask to the cropped depth image
        filtered_image = np.where(mask == 1, depth_image, 0)    

        return filtered_image 


    def visualize_segmented_image(self, rgb_image, mask):
        """
        Overlays the segmentation mask on the RGB image and publishes it to a ROS topic.
        
        Parameters:
        - rgb_image: The original RGB image (numpy array).
        - mask: The binary segmentation mask (numpy array with 0 and 1 values).
        """
        # Convert mask to a 3-channel image (so it can be blended with the RGB image)
        mask_color = np.zeros_like(rgb_image)
        mask_color[:, :, 1] = mask * 255  # Apply the mask to the green channel (you can change the color)
        
        # Blend the original RGB image with the colored mask
        alpha = 0.5  # Transparency factor (can be adjusted)
        blended_image = cv2.addWeighted(rgb_image, 1 - alpha, mask_color, alpha, 0)

        # Convert the blended OpenCV image to a ROS Image message
        segmented_image_msg = self.bridge.cv2_to_imgmsg(blended_image, encoding="bgr8")

        # Publish the segmented image to a ROS topic
        self.segmented_image_publisher.publish(segmented_image_msg)

    
    def visualize_mask_on_depth(self, _resized_mask, depth_image):
        """
        This function overlays the resized mask on top of the depth image for visualization in RViz.
        """
        # Normalize depth image to 8-bit for visualization purposes
        depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = np.uint8(depth_normalized)

        # Apply a colormap to the depth image for better visibility (optional)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

        # Initialize mask_color as a 3-channel image (RGB) with the same shape as depth_colored
        mask_color = np.zeros_like(depth_colored)  # Initialize a 3-channel image for the mask

        # Apply the mask color channel-wise (assign white [255, 255, 255] to all channels where the mask is 1)
        mask_color[_resized_mask == 1] = [255, 255, 255]  # White color for the mask (RGB: [255, 255, 255])

        # Blend depth image and mask for visualization
        _blended_image = cv2.addWeighted(depth_colored, 0.7, mask_color, 0.3, 0)

        # Convert the blended image to a ROS message and publish
        mask_on_depth_msg = self.bridge.cv2_to_imgmsg(_blended_image, encoding="bgr8")
        self.mask_on_depth_publisher.publish(mask_on_depth_msg)
    

    def publish_filtered_image(self, image, topic_name):
        """
        Publishes the given image to a specified ROS topic.
        """
        self.get_logger().info(f"Publishing filtered image on {topic_name}...")  # Debug log
        filtered_image_msg = self.bridge.cv2_to_imgmsg(image, encoding="passthrough")

        if topic_name == "/vidal_filtered_depth_image":
            self._filtered_image_publisher.publish(filtered_image_msg)
        elif topic_name == "/filtered_depth_image":
            self.filtered_image_publisher.publish(filtered_image_msg)
        elif topic_name == "/cropped_image_to_depth":
            self.cropped_image_to_depth_publisher.publish(filtered_image_msg)

        
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
        _filtered_image = np.where(mask, depth_image, 0)

        # Convert _filtered depth image to ROS message and publish
        # _filtered_image_msg = self.bridge.cv2_to_imgmsg(_filtered_image, encoding="passthrough")
        # self._filtered_image_publisher.publish(_filtered_image_msg)

        return _filtered_image


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
    
    # New function to calculate orientation using PCA
    def calculate_orientation_from_bbox_pca(self, x, y, z):
        """
        Calculate the orientation of the 3D bounding box using PCA.
        Args:
            x, y, z: Coordinates of points inside the bounding box.
        Returns:
            quaternion: Orientation quaternion of the object.
        """
        # Stack the points into a single array for PCA
        points = np.column_stack((x, y, z))

        # Perform PCA to find the principal axes
        pca = PCA(n_components=3)
        pca.fit(points)

        # The first principal component gives the major axis (dominant direction of the object)
        direction = pca.components_[0]

        # Calculate the yaw angle (rotation around the Y-axis)
        yaw = np.arctan2(direction[2], direction[0])  # XZ plane

        # Convert yaw into a quaternion (we assume no roll/pitch, so it's just yaw)
        quaternion = tf.quaternion_from_euler(0, yaw, 0)

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
        
        # Calculate orientation and time it
        start_time_orientation = time.time()
        # quaternion = self.calculate_orientation_from_bbox_pca(x, y, z)
        # quaternion =  self.calculate_orientation_from_bbox(min_x, z_min_x, x_min_z, min_z)
        orientation_time = time.time() - start_time_orientation
        self.get_logger().info(f"Orientation Calculation Time: {orientation_time:.4f} seconds")
        self.orientation_time_publisher.publish(Float32(data=orientation_time))
        pose.orientation.x = 0.0 #quaternion[0]
        pose.orientation.y = 0.0 #quaternion[1]
        pose.orientation.z = 0.0 #quaternion[2]
        pose.orientation.w = 1.0 #quaternion[3]
        
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
    node = YoloSAMFilterSegmentationNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
