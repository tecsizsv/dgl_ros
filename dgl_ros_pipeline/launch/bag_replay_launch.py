import os
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    RegisterEventHandler,
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

    rosbag_name_arg = DeclareLaunchArgument('rosbag_name', default_value='rosbag_comp_bottle', description='Name of the rosbag to play')
    bag_play = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play",
            #"rosbag_comp_bottle",
            LaunchConfiguration('rosbag_name'),
            "--loop",
            "--rate", "0.3"
        ],
        output="screen",
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

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(rviz_config_dir, "config", "config.rviz")],
        output="screen",
    )

    gpd_node = Node(
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
    # gpd_node = TimerAction( # action szerver, ami feliratkozik a kamera node által publikált pointclouid topicra, ésa kiszámolt grasp listát az action eredményeként adja vissza, amikor lekérdezik
    #     period=3.0,   # seconds — give the camera time to publish
    #     actions=[
    #         Node(
    #             package="dgl_ros_models",
    #             executable="gpd",
    #             name="gpd",
    #             parameters=[
    #                 {
    #                     "src_topic0": "/camera/camera/depth/color/points",
    #                     "gpd_config_path": "src/dgl_ros/dgl_ros_models/config/gpd_config.yaml",
    #                     "world_frame": "world",
    #                     "src_frame0": "camera_depth_optical_frame",
    #                 }
    #             ],
    #             output="screen",
    #         )
    #     ],
    # )
    return LaunchDescription([
        rosbag_name_arg,
        bag_play,
        static_tf,
        rviz2,
        # gpd_node,
        RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=bag_play,
                on_start=[
                    LogInfo(msg="Bag file started playing..."),
                    TimerAction(period=15.0, actions=[gpd_node]),
                ],
            )
        ),  
    ])