from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Launch human state detector and hand up goal monitor (거리 모니터링만)"""
    
    return LaunchDescription([
        # Human State Detector Node
        Node(
            package='human_utils',
            executable='human_state_detector_node',
            name='human_state_detector',
            output='screen',
            parameters=[{
                'image_topic': '/camera/camera/color/image_raw',
                'camera_info_topic': '/camera/camera/color/camera_info',
                'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
                'camera_frame': 'camera_color_optical_frame',
                'world_frame': 'base',
                'yolo_model': 'yolov8s-pose.pt',
                'yolo_conf': 0.5,
                'yolo_device': 'cuda',
                'publish_markers': True,
                'min_keypoint_conf': 0.3,
            }],
        ),
        
        # Hand Up Goal Monitor Node (모니터링 버전)
        Node(
            package='human_utils',
            executable='hand_up_goal_monitor_node',
            name='hand_up_goal_monitor',
            output='screen',
            parameters=[{
                'map_frame': 'map',
                'goal_distance': 1.2,
                'min_skeleton_points': 5,
                'stop_distance': 0.8,
                'distance_check_rate': 2.0,
                'use_nav2': False,                'min_detection_frames': 5,
                'min_person_distance': 1.5,
                'max_person_distance': 10.0,            }],
        ),

        Node(
            package='human_utils',
            executable='save_current_pose_node',
            name='robot_pose_map_publisher',
            output='screen',
            parameters=[{
                'map_frame': 'map',
                'base_frame': 'base',
                'output_topic': '/robot_pose_map',
                'publish_hz': 10.0,
            }],
        )
    ])
