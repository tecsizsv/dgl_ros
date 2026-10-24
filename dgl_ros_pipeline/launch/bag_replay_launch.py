import os
import yaml
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    RegisterEventHandler,
    LogInfo,)
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import  OnProcessStart
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # Ez a telepített (share) mappára mutat, nem a forrásmappára
    rviz_config_dir = get_package_share_directory('dgl_ros_pipeline')
    gpd_config_dir = get_package_share_directory('dgl_ros_models')
    pipeline_params= os.path.join(get_package_share_directory('dgl_ros_pipeline'), 'config', 'pipeline_params.yaml')
    with open(pipeline_params) as f:
        params = yaml.safe_load(f)
    cam = params['camera_base_tf']['ros__parameters']

    kinematics_file = os.path.join(get_package_share_directory("ur_moveit_config"), "config", "kinematics.yaml")
    with open(kinematics_file) as f:
        kinematics = yaml.safe_load(f)

    rosbag_name_arg = DeclareLaunchArgument('rosbag_name', default_value='cam_aligned_mug2', description='Name of the rosbag to play')
    bag_play = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play",
            #"rosbag_comp_bottle",
            LaunchConfiguration('rosbag_name'),
            "--loop",
            "--rate", "0.3",
            "--progress-bar-update-rate", "0"
        ],
        output="screen",
    )

    static_tf = Node( # publikálja a transzformot: cameralink hol van a world-höz képest
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_base_tf",
        arguments=[
            "--x", str(cam['x']),
            "--y", str(cam['y']),
            "--z", str(cam['z']),
            "--roll", str(cam['roll']),
            "--pitch", str(cam['pitch']),
            "--yaw", str(cam['yaw']),
            "--frame-id", "world",
            "--child-frame-id", "camera_link",
        ],
    )

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory("ur_moveit_config"), "launch", "ur_moveit.launch.py")),
        launch_arguments={"ur_type": "ur5e", "launch_rviz": "false"}.items(),
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(rviz_config_dir, "config", "config.rviz")],
        parameters=[{"robot_description_kinematics": kinematics}],
        output="screen",
    )

    object_detection = Node( # elindítja az object detectiont, ami a kamera topicjára feliratkozik, és a detektált objektumokat publikálja
        package="dgl_ros_pipeline",
        executable="object_detector",
        name="object_detector",
        output="screen",
        parameters=[pipeline_params],
    )

    gpd_node = Node(
        package="dgl_ros_models",
        executable="gpd",
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

    grasp_client = Node( # elindítja a grasp clientet
        package="dgl_ros_pipeline",
        executable="grasp_client",
        name="grasp_client",
        output="screen",
        parameters=[pipeline_params],
    )
    
    return LaunchDescription([
        rosbag_name_arg,
        bag_play,
        static_tf,
        moveit,
        rviz2,
        object_detection,
        RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=bag_play,
                on_start=[
                    LogInfo(msg="Bag file started playing..."),
                    TimerAction(period=15.0, actions=[gpd_node]),
                ],
            )
        ),
        grasp_client,
    ])