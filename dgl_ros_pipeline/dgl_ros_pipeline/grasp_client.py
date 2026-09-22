# Added some notes based on the ROS tutorials (https://docs.ros.org/en/kilted/Tutorials/)
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from rosidl_runtime_py.convert import message_to_yaml
from dgl_ros_interfaces.action import SampleGraspPoses


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
    #end of __init__

    #Method waits for the action server to be available, then sends a goal to the server. It returns a future that we can later wait on.
    def send_goal(self):
        #This method waits for the action server to be available, then sends a goal to the server. It returns a future that we can later wait on.
        goal_msg = SampleGraspPoses.Goal()
        goal_msg.action_name =  'sample_grasp_poses'

        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        self.get_logger().info('Sending goal request...')

        # Version 1: without result and feedback callbacks:
        #return self._action_client.send_goal_async(goal_msg)
        
        # Version 2: without result callbacks (no feedback callbacks):
        #self._send_goal_future = self._action_client.send_goal_async(goal_msg)

        # Version 3: with result and feedback callbacks:      
        self._send_goal_future = self._action_client.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)

        self._send_goal_future.add_done_callback(self.goal_response_callback)
    #end of send_goal

    #Method is called when the action server responds to the goal request. It checks if the goal was accepted or rejected.
    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')

        # self._get_result_future = goal_handle.get_result_async()
        # self._get_result_future.add_done_callback(self.get_result_callback)
    #end of goal_response_callback

    #Method is called when the action server sends the result of the goal.
    # def get_result_callback(self, future):
    #     result = future.result().result
    #     self.get_logger().info('Result: {0}'.format(result.grasp_state))
    #     rclpy.shutdown()
    #end of get_result_callback

    #This method is called when the action server sends feedback about the goal.
    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        #self.get_logger().info('Received feedback: {0}'.format(feedback.grasp_candidates))
        self.get_logger().info(message_to_yaml(feedback))
        # Publish grasp markers for visualization in RViz.
        self.publish_grasp_markers(feedback.grasp_candidates)
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
        
        
def main(args=None):
    try:
        with rclpy.init(args=args):
            # instance of our GraspClient node
            action_client = GraspClient()

            #future = action_client.send_goal('sample_grasp_poses')
            action_client.send_goal()

            #TODO: handle the result of the action (now we only get data in feedback)
            #rclpy.spin_until_future_complete(action_client, future)
            rclpy.spin(action_client)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()