import cv_bridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ultralytics import YOLO

from std_msgs.msg import String


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')
        self.publisher_ = self.create_publisher(String, 'topic', 10) #TODO
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    try:
        with rclpy.init(args=args):
            object_detector = ObjectDetector()

            rclpy.spin(object_detector)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()