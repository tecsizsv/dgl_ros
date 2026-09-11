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

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py",
            )
        ),
        launch_arguments={
            "depth_module.depth_profile": "1280x720x30",
            "pointcloud.enable": "true",
        }.items(),
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_base_tf",
        arguments=[
            # x    y    z  roll pitch yaw   parent   child
            "0", "0", "1.95", "0", "1.5707963", "0",
            "world",
            "camera_link",
        ],
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", "config/rviz_config.rviz"],
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
        realsense_launch,
        static_tf,
        rviz2,
        gpd_node,
    ])