from email.mime import message

import cv_bridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ultralytics import YOLO
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs_py import point_cloud2


class ObjectDetector(Node):
    """Run YOLO detection on ROS2 image messages."""

    def __init__(self):
        """Initialize the ROS2 node, model, and image interfaces."""
        super().__init__('object_detector')
        self.bridge = cv_bridge.CvBridge()
        self.model = YOLO("yolo26m.pt")

        self.publisher = self.create_publisher(Image, "/ultralytics/detection/image", 5)
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.callback, qos_profile_sensor_data)

        self.create_subscription(Image, "/camera/camera/color/image_raw", self.rgb_callback, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/camera/depth/image_raw", self.depth_callback, qos_profile_sensor_data)

    def rgb_callback(self, message):
        self.rgb_image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

    def depth_callback(self, message):
        self.depth_image = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
    # Apply the NumPy mask and distance calculation from the depth example below.    


    def callback(self, message):
        """Publish the annotated camera frame."""
        image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        annotated = self.model(image)[0].plot(show=False)
        self.publisher.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))
        points = point_cloud2.read_points_numpy(message, field_names=("x", "y", "z", "rgb"))
        points = points.reshape(message.height, message.width, 4)


def main(args=None):
    # try:
    #     with rclpy.init(args=args):
    #         object_detector = ObjectDetector()

    #         rclpy.spin(object_detector)
    # except (KeyboardInterrupt, ExternalShutdownException):
    #     pass

    """Start the ROS2 node."""
    rclpy.init(args=args)
    node = ObjectDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()