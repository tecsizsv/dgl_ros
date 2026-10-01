import cv_bridge
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from ultralytics import YOLO
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from message_filters import Subscriber, ApproximateTimeSynchronizer

# qos_profile_sensor_data: Quality of Service (QoS) profile

class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        self._bridge = cv_bridge.CvBridge()

        # YOLO model
        self._model = YOLO("yolo26m.pt")

        queue_size = 10
        max_delay = 0.05

        # Subscribers:
        #Pointcloud
        # self._pointcloud_sub = self.create_subscription(PointCloud2, "/camera/camera/depth/color/points", self.pointcloud_callback, qos_profile_sensor_data)
        
        # #RGB image        
        self._rgb_sub = self.create_subscription(Image, "/camera/camera/color/image_raw", self.rgb_callback, qos_profile_sensor_data)
        #self._rgb_sub = Subscriber(self, Image, "/camera/camera/color/image_raw", qos_profile_sensor_data)

        # #Depth image
        # #self._depth_sub = self.create_subscription(Image, "/camera/camera/depth/image_raw", self.depth_callback, qos_profile_sensor_data)
        # self._depth_sub = Subscriber(self, Image, "/camera/camera/depth/image_raw", qos_profile_sensor_data)

        # self.time_sync = ApproximateTimeSynchronizer([self._rgb_sub, self._depth_sub], queue_size, max_delay)  
        # self.time_sync.registerCallback(self.sync_callback)

        # Publishers:
        #annotated image
        self._detection_image_pub = self.create_publisher(Image, "detection_image", 10)
        #bounding box and object name
        #self._detections_pub = self.create_publisher(Image, "detections", 10)
    
    #end of __init__

    # def pointcloud_callback(self, message):

    #     points = point_cloud2.read_points_numpy(message, field_names=("x", "y", "z", "rgb"))
    #     points = points.reshape(message.height, message.width, 4)



    #     # self._model(image): run YOLO on the image (result is an array/list)
    #     # .plot(show=false): draws the bounding box on the image
    #     # show=false: don't open display window
    #     result = self._model(points)[0].plot(show=False)

    #     # Publishing the result
    #     self._detections_pub.publish(result)
    # #end of pointcloud_callback

    def rgb_callback(self, message):
        # Convert ROS Image message to OpenCV image
        self._rgb_image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")

        # Run YOLO model on the RGB image
        detection_image = self._model(self._rgb_image)[0].plot(show=False)

        # Publish the annotated image
        self._detection_image_pub.publish(self._bridge.cv2_to_imgmsg(detection_image, encoding="bgr8"))
    #end of rgb_callback


    # def sync_callback(self, rgb_msg, depth_msg):
    #     self._rgb_image = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
    #     self._depth_image = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")

    #     detection_image = self._model(self._rgb_image)[0].plot(show=False)

    #     self._detections_pub.publish(self._bridge.cv2_to_imgmsg(detection_image, encoding="bgr8"))
    # # #end of sync_callback


def main(args=None):
    try:
        with rclpy.init(args=args):
            object_detector = ObjectDetector()

            rclpy.spin(object_detector)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()