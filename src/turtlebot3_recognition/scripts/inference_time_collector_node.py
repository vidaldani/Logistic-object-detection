#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
import pandas as pd
from datetime import datetime
# import time

class InferenceTimeCollectorNode(Node):

    def __init__(self):
        super().__init__('inference_time_collector_node')

        # Initialize inference time data
        self.data = []
        self.detection_count = 0
        self.max_detections = 100

        # Variables to store individual stage times for each detection
        self.current_detection = {
            'YOLO Inference': None,
            'Depth Filtering': None,
            'SAM Segmentation': None,
            'Orientation Calculation': None,
        }


        # Flags to determine if SAM time is expected
        self.sam_enabled = False

        # Subscribe to YOLO inference start and end times (from yolov8_ros2_pt node)
        self.yolo_subscriber = self.create_subscription(
            Float32,
            '/yolo_inference_time',
            self.yolo_inference_callback,
            10
        )

        # Subscribe to Depth Filtering time (from depth segmentation node or SAM node)
        self.depth_filtering_subscriber = self.create_subscription(
            Float32,
            '/depth_filtering_time',
            self.depth_filtering_callback,
            10
        )

        # Subscribe to SAM Segmentation time
        self.sam_segmentation_subscriber = self.create_subscription(
            Float32,
            '/sam_segmentation_time',
            self.sam_segmentation_callback,
            10
        )

        # Subscribe to Orientation calculation time
        self.orientation_time_subscriber = self.create_subscription(
            Float32,
            '/orientation_time',
            self.orientation_callback,
            10
        )

        self.get_logger().info("InferenceTimeCollectorNode initialized.")

    def yolo_inference_callback(self, msg):
        # Log the YOLO inference time
        self.current_detection['YOLO Inference'] = msg.data
        self.log_and_check_if_complete()

    def depth_filtering_callback(self, msg):
        # Log the Depth Filtering time
        self.current_detection['Depth Filtering'] = msg.data
        self.log_and_check_if_complete()

    def sam_segmentation_callback(self, msg):
        # Log the SAM Segmentation time
        self.current_detection['SAM Segmentation'] = msg.data
        self.sam_enabled = True  # Set flag to indicate SAM is part of the pipeline
        self.log_and_check_if_complete()

    def orientation_callback(self, msg):
        # Log the Orientation Calculation time
        self.current_detection['Orientation Calculation'] = msg.data
        self.log_and_check_if_complete()

    def log_and_check_if_complete(self):
        # Check if all stages are completed for a single detection
        # if all(self.current_detection.values()) and self.detection_count < self.max_detections:
        #     total_pipeline_time = sum(self.current_detection.values())
        
        # Check if all mandatory stages (YOLO, Depth Filtering, Orientation) are completed
        if (self.current_detection['YOLO Inference'] is not None and
            self.current_detection['Depth Filtering'] is not None and
            self.current_detection['Orientation Calculation'] is not None):

            # If SAM is enabled, ensure SAM time is also available
            if self.sam_enabled and self.current_detection['SAM Segmentation'] is None:
                return  # Wait for SAM time if it's part of the pipeline

            # Calculate total pipeline time (with or without SAM)
            total_pipeline_time = (
                self.current_detection['YOLO Inference'] +
                self.current_detection['Depth Filtering'] +
                (self.current_detection['SAM Segmentation'] if self.sam_enabled else 0) +
                self.current_detection['Orientation Calculation']
            )

            # Calculate the frames per second (FPS) based on the total pipeline time
            if total_pipeline_time > 0:
                fps = 1 / total_pipeline_time
            else:
                fps = float('inf')  # Handle case if total pipeline time is 0 (unlikely)

            # Append the data for this detection
            self.data.append({
                'YOLO Inference': self.current_detection['YOLO Inference'],
                'Depth Filtering': self.current_detection['Depth Filtering'],
                'SAM Segmentation': self.current_detection['SAM Segmentation'],
                'Orientation Calculation': self.current_detection['Orientation Calculation'],
                'Total Pipeline Time': total_pipeline_time,
                'Total Pipeline Time in FPS': fps
            })

            self.detection_count += 1
            self.get_logger().info(f'Detection {self.detection_count} recorded with total pipeline time: {total_pipeline_time:.4f} seconds')

            # Reset the current detection times
            self.current_detection = {
                'YOLO Inference': None,
                'Depth Filtering': None,
                'SAM Segmentation': None,
                'Orientation Calculation': None,
            }

            # Save and shutdown after 100 detections
            if self.detection_count == self.max_detections:
                self.save_to_excel()
                self.get_logger().info('Collected 100 detections, shutting down.')
                rclpy.shutdown()

    def save_to_excel(self):
        # Convert data to DataFrame
        df = pd.DataFrame(self.data)

        # Generate the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Format to display only 3 decimal places in Excel (without affecting the internal precision)
        # pd.options.display.float_format = '{:.3f}'.format

        # Save to Excel file
        file_name = f'inference_times_{timestamp}_SAM2b+_Forklift_no_or.xlsx'
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
            max_width = 26   # Maximum width to avoid excessively wide columns for numbers

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
    node = InferenceTimeCollectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
