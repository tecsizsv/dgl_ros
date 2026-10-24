import os
import yaml
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    ExecuteProcess,
    RegisterEventHandler,)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import OnProcessExit
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

    realsense_launch = IncludeLaunchDescription( # elindítja a kamerát, ami a pointcloud topicokat publikálja
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("realsense2_camera"),
                "launch",
                "rs_launch.py",
            )
        ),
        launch_arguments={
            "json_file_path": os.path.join(get_package_share_directory("dgl_ros_pipeline"), "config", "camera_config2.json",),
            "config_file": os.path.join(get_package_share_directory("dgl_ros_pipeline"), "config", "camera_filters.yaml"),
        }.items(),
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

    rviz2 = Node( # elindítja az rviz-t
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(rviz_config_dir, "config", "config.rviz")],
        output="screen",
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

    object_detection = Node( # elindítja az object detectiont, ami a kamera topicjára feliratkozik, és a detektált objektumokat publikálja
        package="dgl_ros_pipeline",
        executable="object_detector",
        name="object_detector",
        output="screen",
        parameters=[pipeline_params],
    )

    grasp_client = Node( # elindítja a grasp clientet
        package="dgl_ros_pipeline",
        executable="grasp_client",
        name="grasp_client",
        output="screen",
        parameters=[pipeline_params],
    )

    wait_for_camera = ExecuteProcess(
        cmd=["ros2", "topic", "echo", "--once", "/camera/camera/depth/color/points"],
        output="log",
    )

    return LaunchDescription([
        wait_for_camera,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=wait_for_camera,
                on_exit=[gpd_node],
            )
     ),
        realsense_launch,
        static_tf,
        moveit,
        rviz2,
        object_detection,
        grasp_client,
    ])