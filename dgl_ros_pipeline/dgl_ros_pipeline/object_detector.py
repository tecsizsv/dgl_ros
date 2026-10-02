import cv_bridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ultralytics import YOLO
from ultralytics import YOLOWorld
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from message_filters import Subscriber, ApproximateTimeSynchronizer

# qos_profile_sensor_data: Quality of Service (QoS) profile

class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        self._bridge = cv_bridge.CvBridge()

        # Loading YOLO model
        #self._model = YOLO("yolo26m-objv1-150.pt") # Objects 365 data set
        #self._model = YOLO("yolo26m.pt") # COCO data set
        self._model = YOLOWorld("yolov8s-world.pt") # World data set
        self._model.set_classes(["cardboard box", "battery", "bottle", "rod"])

        #self._model.task = "detect"
    
        # Subscribers:
        self.create_subscription(Image, "/camera/camera/color/image_raw", self.image_callback, qos_profile_sensor_data)

        # Publishers:
        self._detection_pub = self.create_publisher(Image, "detections", 10)

    
    #end of __init__

    def image_callback(self, message):
        # Convert ROS Image message to OpenCV image
        image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        #image[y1:y2, x1:x2]
        cropped = image[20:720, 440:1090]

        # Run YOLO model on the converted image
        detections = self._model(cropped)[0].plot(show=False)
        #detections = self._model.predict(cropped)[0].plot(show=False)
        self.get_logger().info("Object detection completed.")

        # Publish the annotated image
        self._detection_pub.publish(self._bridge.cv2_to_imgmsg(detections, encoding="bgr8"))
    #end of image_callback

0
def main(args=None):
    try:
        with rclpy.init(args=args):
            object_detector = ObjectDetector()

            rclpy.spin(object_detector)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()