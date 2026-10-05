# Added some notes based on the ROS tutorials (https://docs.ros.org/en/kilted/Tutorials/)
import time
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from rosidl_runtime_py.convert import message_to_yaml
from dgl_ros_interfaces.action import SampleGraspPoses
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from action_msgs.msg import GoalStatus
from sensor_msgs.msg import CameraInfo

class GraspClient(Node):

    def __init__(self):
        # The class is initialized by calling the Node constructor, naming our node grasp_client:
        super().__init__('grasp_client')

        # Creating an action client that will send goals to the SampleGraspPoses action server.
        # Three arguments:
        #   1. A ROS 2 node to add the action client to: self (aka. GraspClient)
        #   2. The type of the action: SampleGraspPoses
        #   3. The action name: 'sample_grasp_poses'
        self._action_client = ActionClient(self, SampleGraspPoses, 'sample_grasp_poses')

        # Creating a publisher for visualization markers.
        self._marker_pub = self.create_publisher(MarkerArray, 'grasp_markers', 10)

        # Subscriber to YOLO detection messages (Detection2DArray)
        self._det_sub = self.create_subscription(Detection2DArray, 'detections', self.detection_callback, 10)

        # Flags to controll actions
        self.busy = False

        #---Some parameters from K matrix of Camera---
        # Can read it from /camera/camera/color/camera_info
        #camera focal lenght in pixel
        self.fx = 922.5073852539062
        self.fy = 920.6383056640625
        # optical center of the image in pixel
        self.cx = 645.260498046875 
        self.cy = 354.32769775390625
        # camera distance from the table in meter
        self.cam_to_table = 1.95
        # Workspace boundaries on Z-axis in meter
        self.z_min = 0.0
        self.z_max = 1.75
        # How much bigger should the workspace be than the boudingbox in meter
        self.margin = 0.05 # 5 cm
    #end of __init__

    #Method waits for the action server to be available, then sends a goal to the server. It returns a future that we can later wait on.
    def send_goal(self, workspace):
        goal_msg = SampleGraspPoses.Goal()
        goal_msg.action_name =  'sample_grasp_poses'
        goal_msg.workspace = workspace

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info('Sending goal request...')

        # Async callback      
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)

    #end of send_goal

    #Method is called when the action server responds to the goal request. It checks if the goal was accepted or rejected.
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            self.busy = False
            return

        self.get_logger().info('Goal accepted, waiting for result :)')
        
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)
    #end of goal_response_callback

    #Method is called when the action server sends the result of the goal.
    def get_result_callback(self, future):
        result = future.result().result

        self.get_logger().info(message_to_yaml(result))

        self.publish_grasp_markers(result.grasp_candidates)

        self.busy = False
        #Alternative solution with some error handling
        # try:
        #     result = future.result().result
        #     status = future.result().status
        #     if status == GoalStatus.STATUS_SUCCEEDED:
        #         self.get_logger().info('Goal succeeded!')
        #         self.get_logger().info(message_to_yaml(result))
        #         self.publish_grasp_markers(result.grasp_candidates)
        #     else:
        #         self.get_logger().info('Goal failed with status: {0}'.format(status))
        # finally:
        #     self.busy = False
    #end of get_result_callback

    #This method is called when the action server sends feedback about the goal.
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback

        self.get_logger().info(message_to_yaml(feedback))
    #end of feedback_callback

    #This method creates a marker for a grasp, it's a tool used only inside this class
    def _make_marker(self, grasp_candidate, marker_id):
        #start_point = grasp_candidate.pose.position - hand_depth * grasp_candidate.approach
        #end_point = grasp_candidate.pose.position
        marker = Marker()
        marker.header.frame_id = grasp_candidate.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()

        # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
        marker.type = 0
        marker.id = marker_id
        marker.ns = 'grasp_candidates'

        marker.action = Marker.ADD

        #marker.points = [start_point, end_point]

        # Set the scale of the marker
        marker.scale.x = 0.2 # length
        marker.scale.y = 0.03 # width
        marker.scale.z = 0.03 # height

        # Set the color
        #pretty pink: 0.922, 0.114, 0.596
        #pretty green: 0.573, 0.961, 0.584
        marker.color.r = 0.922
        marker.color.g = 0.114
        marker.color.b = 0.596
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = grasp_candidate.pose.position.x
        marker.pose.position.y = grasp_candidate.pose.position.y
        marker.pose.position.z = grasp_candidate.pose.position.z
        marker.pose.orientation.x = grasp_candidate.pose.orientation.x
        marker.pose.orientation.y = grasp_candidate.pose.orientation.y
        marker.pose.orientation.z = grasp_candidate.pose.orientation.z
        marker.pose.orientation.w = grasp_candidate.pose.orientation.w

        return marker
    #end of _make_marker 
  
    #This method is called to publish grasp markers for visualization in RViz.
    def publish_grasp_markers(self, grasp_candidates):
        marker_array = MarkerArray()

        #Delete any remaining markers from previous grasps
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL

        marker_array.markers.append(delete_marker)

        #Create a marker for each grasp candidate and add it to the marker array
        for i in range(len(grasp_candidates)):
            marker = self._make_marker(grasp_candidates[i], i)
            marker_array.markers.append(marker)

        self._marker_pub.publish(marker_array)  
    #end of publish_grasp_markers
        
    def detection_callback(self, msg):
        if self.busy:
            return

        # # Process the incoming detection messages
        # for det in msg.detections:
        #     pos_x = det.bbox.center.position.x
        #     pos_y = det.bbox.center.position.y
        #     size_x = det.bbox.size_x
        #     size_y = det.bbox.size_y

        # Logging bounding box measures for debug purposes
        # self.get_logger().info(
        # f'{det.results[0].hypothesis.class_id} ({det.results[0].hypothesis.score:.2f}) at ({pos_x:.0f},{pos_y:.0f}) '
        # f'size {size_x:.0f}x{size_y:.0f}'
        # )

        # Check if there is a message
        if not msg:
            return
        
        det = msg.detections[0]


        workspace = self._calc_workspace(det.bbox)

        self.busy = True
        self.get_logger().info(f'workspace = {workspace}')
        self.send_goal(workspace)
    #end of detection_callback

    #  This method converts the boundingbox from pixels to meters thus a workspace is calculated
    def _calc_workspace(self, bbox):
        bbox_center_x = bbox.center.position.x
        bbox_center_y = bbox.center.position.y
        bbox_width = bbox.size_x
        bbox_height = bbox.size_y

        # Calculating the sides of the boundingbox
        u_min =  bbox_center_x-(bbox_width/2)
        u_max = u_min + bbox_width
        v_min = bbox_center_y-(bbox_height/2)
        v_max = v_min + bbox_height

        # Converting sides from 2D to 3D: P_cam = K^-1 * p
        # K: 922.5073852539062  0.0                645.260498046875
        #    0.0                920.6383056640625  354.32769775390625
        #    0.0                0.0                1.0
        # p = [u, v, 1]' P_cam = [x, y, z]'  
        depth = self.cam_to_table
        x_min = (u_min - self.cx) * depth / self.fx
        x_max = (u_max - self.cx) * depth / self.fx
        y_min = (v_min - self.cy) * depth/ self.fy
        y_max = (v_max - self.cy) * depth/ self.fy
        z = depth

        # Transform from camera_color_optical_frame to world: P = T * P_cam (adding 1 at the end)
        # T: -0.009 -1.000  0.003  0.000
        #    -1.000  0.009 -0.007  0.015
        #     0.007 -0.003 -1.000  1.950
        #     0.000  0.000  0.000  1.000
        # First calculate with the min values, and after that with max values
        x_a = (-0.009 * x_min - 1 * y_min + 0.003 * z + 0.0) 
        x_b = (-0.009 * x_max - 1 * y_max + 0.003 * z + 0.0) 
        y_a = (-1.0 * x_min + 0.009 * y_min - 0.007 * z + 0.015) 
        y_b = (-1.0 * x_max + 0.009 * y_max - 0.007 * z + 0.015) 

        x_world_min = min(x_a, x_b) - self.margin
        x_world_max = max(x_a, x_b) + self.margin
        y_world_min = min(y_a, y_b) - self.margin
        y_world_max = max(y_a, y_b) + self.margin
        z_world_min = self.z_min
        z_world_max = self.z_max

        workspace = [x_world_min, x_world_max, y_world_min, y_world_max, z_world_min, z_world_max]

        return workspace
    #end of _calc_workspace

def main(args=None):
    try:
        with rclpy.init(args=args):
            # instance of our GraspClient node
            action_client = GraspClient()

            #delay 5 seconds to allow the action server to start up
            time.sleep(5)

            #future = action_client.send_goal()
            #action_client.send_goal()

            rclpy.spin(action_client)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()