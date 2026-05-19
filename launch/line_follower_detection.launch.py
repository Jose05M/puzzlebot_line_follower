import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    pkg = FindPackageShare('puzzlebot_line_follower')

    config_arg = DeclareLaunchArgument(
        'config',
        default_value=PathJoinSubstitution([pkg, 'config', 'params.yaml']),
        description='Path to YAML parameter file',
    )

    config_file = LaunchConfiguration('config')

    detector_node = Node(
        package='puzzlebot_line_follower',
        executable='traffic_light_detector',
        name='traffic_light_detector',
        parameters=[config_file],
        output='screen',
        emulate_tty=True,
    )

    controller_node = Node(
        package='puzzlebot_line_follower',
        executable='line_follower_controller',
        name='line_follower_controller',
        parameters=[config_file],
        output='screen',
        emulate_tty=True,
    )

    odom_node = Node(
        package='puzzlebot_line_follower',
        executable='puzzlebot_odometry',
        name='puzzlebot_odometry',
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        config_arg,
        odom_node,
        detector_node,
        controller_node,
    ])
