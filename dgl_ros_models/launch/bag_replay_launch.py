import os
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    ExecuteProcess,
    TimerAction,
    LogInfo,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    bag_play = ExecuteProcess(
    cmd=[
        "ros2", "bag", "play",
        "rosbag_comp_bottle",
        "--loop",
        "-r", "0.5"
    ],
    output="screen",
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", "config.rviz"],
        output="screen",
    )

    gpd_node = TimerAction(
        period=3.0,   # seconds — give the camera time to publish
        actions=[
            Node(
                package="dgl_ros_models",
                executable="gpd",
                name="gpd",
                parameters=[
                    {
                        "src_topic0": "/camera/camera/depth/color/points",
                        "gpd_config_path": "src/dgl_ros/dgl_ros_models/config/gpd_config.yaml",
                        "world_frame": "world",
                        "src_frame0": "camera_depth_optical_frame",
                    }
                ],
                output="screen",
            )
        ],
    )

    send_grasp_goal = TimerAction(
        period=10.0,  # seconds — wait for GPD to be ready
        actions=[
            LogInfo(msg="Sending SampleGraspPoses action goal..."),
            ExecuteProcess(
                cmd=[
                    "ros2", "action", "send_goal",
                    "/sample_grasp_poses",
                    "dgl_ros_interfaces/action/SampleGraspPoses",
                    "{action_name: 'sample_grasp_poses'}",
                ],
                output="screen",
            ),
        ],
    )

    return LaunchDescription([
        bag_play,
        rviz2,
        gpd_node,
    ])