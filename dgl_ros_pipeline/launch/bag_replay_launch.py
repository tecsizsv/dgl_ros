import os
import yaml
import json
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
    RegisterEventHandler,
    LogInfo,
    IncludeLaunchDescription,)
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.event_handlers import  OnProcessStart, OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    ur_type = "ur5e"
    onrobot_type = "rg2"

    # Share directories
    dgl_share = get_package_share_directory("dgl_ros_pipeline")
    gpd_share = get_package_share_directory("dgl_ros_models")
    onrobot_moveit_share = get_package_share_directory("ur_onrobot_moveit_config")
    onrobot_description_share= get_package_share_directory("ur_onrobot_description")
    
    # Config file paths
    ur_model_config = os.path.join(get_package_share_directory("ur_description"), "config", ur_type)
    pipeline_params= os.path.join(dgl_share, "config", "pipeline_params.yaml")
    kinematics_file = os.path.join(onrobot_moveit_share, "config", "kinematics.yaml")
    controllers_file = os.path.join(onrobot_moveit_share, "config", "controllers.yaml")

    # MoveItPy settings
    moveit_cpp_yaml = os.path.join(dgl_share, "config", "moveit_cpp.yaml")
    
    # Loading params
    with open(pipeline_params) as f:
        params = yaml.safe_load(f)
    cam = params['camera_base_tf']['ros__parameters']

    with open(kinematics_file) as f:
        kinematics = yaml.safe_load(f)

    with open(controllers_file) as f:
        controllers = yaml.safe_load(f)

    rosbag_name_arg = DeclareLaunchArgument('rosbag_name', default_value='cam_aligned_mug2', description='Name of the rosbag to play')
    log_level_arg = DeclareLaunchArgument('log_level', default_value='info')
    
    bag_play = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play",
            LaunchConfiguration('rosbag_name'),
            "--loop",
            "--rate", "0.3",
            "--progress-bar-update-rate", "0"
        ],
        output="screen",
    )

    static_tf = Node( # camera position relative to world frame
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
        PythonLaunchDescriptionSource(os.path.join(onrobot_moveit_share, "launch", "ur_onrobot_moveit.launch.py")),
        launch_arguments={
            "ur_type": ur_type,
            "onrobot_type": onrobot_type,
            "launch_servo": "false",
            "launch_rviz": "false"
        }.items(),
    )

    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(dgl_share, "config", "config.rviz")],
        parameters=[{"robot_description_kinematics": kinematics}],
        ros_arguments=['--log-level', 'warn'],
        output="log",
    )

    object_detection = Node(
        package="dgl_ros_pipeline",
        executable="object_detector",
        name="object_detector",
        output="both",
        parameters=[pipeline_params],
        ros_arguments=['--log-level', LaunchConfiguration('log_level')],
    )

    gpd_node = Node(
        package="dgl_ros_models",
        executable="gpd",
        parameters=[{
                "src_topic0": "/camera/camera/depth/color/points",
                "gpd_config_path": os.path.join(gpd_share, "config", "gpd_config.yaml"),
                "world_frame": "world",
                "src_frame0": "camera_depth_optical_frame",
        }],
        output="log",
    )

    moveit_config = (
        MoveItConfigsBuilder(robot_name="ur_onrobot", package_name="ur_onrobot_moveit_config")
        .robot_description(
            file_path=os.path.join(onrobot_description_share, "urdf", "ur_onrobot.urdf.xacro"),
            mappings={
                "robot_ip": "xxx.yyy.zzz.www",
                "joint_limit_params": os.path.join(ur_model_config, "joint_limits.yaml"),                       
                "kinematics_params": os.path.join(ur_model_config, "default_kinematics.yaml"),
                "physical_params": os.path.join(ur_model_config, "physical_parameters.yaml"),
                "visual_params": os.path.join(ur_model_config, "visual_parameters.yaml"),
                "safety_limits": "true",
                "safety_pos_margin": "0.15",
                "safety_k_position": "20",
                "name": "ur_onrobot",
                "ur_type": ur_type,
                "onrobot_type": onrobot_type,
                "script_filename": "ros_control.urscript",
                "input_recipe_filename": "rtde_input_recipe.txt",
                "output_recipe_filename": "rtde_output_recipe.txt",
                "prefix": ""
            },
        )
        .robot_description_semantic(
            file_path=os.path.join(onrobot_moveit_share, "srdf", "ur_onrobot.srdf.xacro"),
            mappings={"name": "ur_onrobot", "prefix": ""},
        )
        .robot_description_kinematics(file_path=kinematics_file)
        .joint_limits(file_path=os.path.join(onrobot_moveit_share, "config", "joint_limits.yaml"))
        #.trajectory_execution(file_path=os.path.join(onrobot_moveit_share, "config", "controllers.yaml"))
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .moveit_cpp(file_path=os.path.join(dgl_share, "config", "moveit_cpp.yaml"))
        .to_moveit_configs()
    )

    moveit_params = moveit_config.to_dict()

    response_adapters = moveit_params["ompl"].setdefault("response_adapters", [])
    timing_adapter = "default_planning_response_adapters/AddTimeOptimalParameterization"
    display_adapter = "default_planning_response_adapters/DisplayMotionPath"

    if display_adapter not in response_adapters:
        response_adapters.append(display_adapter)

    if timing_adapter in response_adapters:
        response_adapters.remove(timing_adapter)

    response_adapters.insert(response_adapters.index(display_adapter), timing_adapter)

    moveit_params["moveit_controller_manager"] = ("moveit_simple_controller_manager/MoveItSimpleControllerManager")
    moveit_params["moveit_simple_controller_manager"] = controllers
    moveit_params["moveit_manage_controllers"] = False

    #print(yaml.safe_dump(moveit_params["ompl"], sort_keys=False))

    grasp_client = Node(
        package="dgl_ros_pipeline",
        executable="grasp_client",
        name="grasp_client",
        output="both",
        parameters=[
            pipeline_params,
            {"moveit_config_json": json.dumps(moveit_params)},
            {"use_sim_time": False},
        ],
        ros_arguments=[
            '--log-level', 'warn',
            '--log-level', ['grasp_client:=', LaunchConfiguration('log_level')]
        ]
    )

    wait_for_camera = ExecuteProcess(
        cmd=["ros2", "topic", "echo", "--once", "--no-arr",
            "/camera/camera/depth/color/points"],
        output="log",
    )

    start_gpd_after_bag = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=wait_for_camera,
            on_exit=[gpd_node],
        )
    )

    return LaunchDescription([
        rosbag_name_arg,
        log_level_arg,
        bag_play,
        wait_for_camera,
        static_tf,
        moveit,
        rviz2,
        object_detection,
        start_gpd_after_bag,
        grasp_client,
    ])