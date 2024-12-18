from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare the 'working_frame' argument with a default value
    working_frame_arg = DeclareLaunchArgument(
        'working_frame',
        default_value='odom',
        description='The working frame for 3D marker publisher'
    )

    # Define the node to run yolo_sam2_segmentation.py
    yolo_sam2_segmentation_node = Node(
        package='turtlebot3_recognition',  # Replace with your actual package name
        executable='yolo_sam2_segmentation.py',
        name='yolo_sam2_segmentation',
        output='screen',
        emulate_tty=True
    )

    # Define the node to run 3d_marker_publisher.py with the 'working_frame' argument
    marker_publisher_node = Node(
        package='turtlebot3_recognition',  # Replace with your actual package name
        executable='bounding_box_marker_publisher.py',
        name='bounding_box_marker_publisher',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'working_frame': LaunchConfiguration('working_frame')
        }]
    )

    # Create the launch description with both nodes
    return LaunchDescription([
        working_frame_arg,
        yolo_sam2_segmentation_node,
        marker_publisher_node
    ])
