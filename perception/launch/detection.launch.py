from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def launch_setup(context, *args, **kwargs):
    image_topic_name = LaunchConfiguration('image_topic_name')
    image_transport = LaunchConfiguration('image_transport')
    camera_info_topic_name = LaunchConfiguration('camera_info_topic_name')
    depth_topic_name = LaunchConfiguration('depth_topic_name')
    enable_pointcloud = LaunchConfiguration('enable_pointcloud')
    enable_depth_ema = LaunchConfiguration('enable_depth_ema')
    search_timeout = LaunchConfiguration('search_timeout')

    yolo_node = Node(
        package='perception',
        executable='yolo_node',
        name='yolo_node',
        output='screen',
        parameters=[{
            'image_topic': image_topic_name,
            'image_type': image_transport,
        }]
    )
    groundedsam2_node = Node(
        package='perception',
        executable='groundedsam2_node',
        name='groundedsam2_node',
        output='screen',
        parameters=[{
            'image_topic': image_topic_name,
            'image_type': image_transport,
            'camera_info_topic': camera_info_topic_name,
            'enable_pointcloud': enable_pointcloud,
            'search_timeout': search_timeout,
        }]
    )
    depth_ema_node = Node(
        package='perception',
        executable='depth_ema_node',
        name='depth_ema_node',
        output='screen',
        parameters=[{
            'depth_topic': depth_topic_name,
        }],
        condition=IfCondition(enable_depth_ema)
    )

    return [depth_ema_node, yolo_node, groundedsam2_node]


def generate_launch_description():
    image_topic_name_arg = DeclareLaunchArgument(
        'image_topic_name',
        default_value='/camera/camera/color/image_raw',
        description='Image topic name to subscribe to'
    )
    image_transport_arg = DeclareLaunchArgument(
        'image_transport',
        default_value='raw',
        description='Image transport type (raw or compressed)'
    )
    camera_info_topic_name_arg = DeclareLaunchArgument(
        'camera_info_topic_name',
        default_value='/camera/camera/color/camera_info',
        description='Camera info topic name to subscribe to'
    )
    depth_topic_name_arg = DeclareLaunchArgument(
        'depth_topic_name',
        default_value='/camera/camera/aligned_depth_to_color/image_raw',
        description='Depth topic name to subscribe to'
    )
    enable_pointcloud_arg = DeclareLaunchArgument(
        'enable_pointcloud',
        default_value='true',
        description='Enable pointcloud generation'
    )
    enable_depth_ema_arg = DeclareLaunchArgument(
        'enable_depth_ema',
        default_value='true',
        description='Enable Depth EMA filtering'
    )
    search_timeout_arg = DeclareLaunchArgument(
        'search_timeout',
        default_value='2.0',
        description='Timeout for searching objects in seconds'
    )

    # turn_on_python_venv = ExecuteProcess(
    #     cmd=[[
    #         'source',
    #         '/home/nvidia/vision_ws/src/perception/venv/bin/activate'
    #         ]],
    #     shell=True
    # )
    
    # turn_on_install = ExecuteProcess(
    #     cmd=[[
    #         'source',
    #         '/home/nvidia/vision_ws/install/setup.bash'
    #         ]],
    #     shell=True
    # )

    return LaunchDescription([
        image_topic_name_arg,
        image_transport_arg,
        camera_info_topic_name_arg,
        depth_topic_name_arg,
        enable_pointcloud_arg,
        enable_depth_ema_arg,
        search_timeout_arg,
        OpaqueFunction(function=launch_setup),
    ])
