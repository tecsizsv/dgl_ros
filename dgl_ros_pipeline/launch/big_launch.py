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
from launch.event_handlers import (
    OnExecutionComplete,
    OnProcessExit,
    OnProcessIO,
    OnProcessStart,
    OnShutdown
)

def generate_launch_description():

    # Ez a telepített (share) mappára mutat, nem a forrásmappára
    rviz_config_dir = get_package_share_directory('dgl_ros_pipeline')
    gpd_config_dir = get_package_share_directory('dgl_ros_models')

    realsense_launch = IncludeLaunchDescription( # elindítja a kamerát, ami a pointcloud topicokat publikálja
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

    static_tf = Node( # publikálja a transzformot: cameralink hol van a world-höz képest
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

    rviz2 = Node( # elindítja az rviz-t
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(rviz_config_dir, "config", "config.rviz")],
        output="screen",
    )

    gpd_node = TimerAction( # action szerver, ami feliratkozik a kamera node által publikált pointclouid topicra, ésa kiszámolt grasp listát az action eredményeként adja vissza, amikor lekérdezik
        period=3.0,   # seconds — give the camera time to publish
        actions=[
            Node(
                package="dgl_ros_models",
                executable="gpd",
                name="gpd",
                parameters=[
                    {
                        "src_topic0": "/camera/camera/depth/color/points",
                        "gpd_config_path": os.path.join(gpd_config_dir, "config", "gpd_config.yaml"),
                        "world_frame": "world",
                        "src_frame0": "camera_depth_optical_frame",
                    }
                ],
                output="screen",
            )
        ],
    )

    send_grasp_goal = TimerAction( # ez a hívás elindítja a grasp-számítást, nem egy már kész eredményt kér le
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