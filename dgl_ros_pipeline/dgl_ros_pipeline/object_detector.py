import cv2
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from rclpy.qos import qos_profile_sensor_data
import cv_bridge
from ultralytics import YOLO
from ultralytics import YOLOWorld
import numpy as np

# qos_profile_sensor_data: Quality of Service (QoS) profile
CROP_X_OFFSET = 440
CROP_Y_OFFSET = 20

# Color palette, one color per class_id
PALETTE = np.random.randint(0, 255, (80, 3), dtype=np.uint8)

class ObjectDetector(Node):

    def __init__(self):
        super().__init__('object_detector')

        self._bridge = cv_bridge.CvBridge()

        # Loading YOLO model
        #self._model = YOLO("yolo26m-objv1-150.pt") # Objects 365 data set
        #self._model = YOLO("yolo26m.pt") # COCO data set
        #self._model = YOLOWorld("yolov8s-world.pt") # World data set
        self._model = YOLO("yolov8s-world.pt") # World data set
        self._model.set_classes(["cardboard box", "battery", "bottle", "rod"])

    
        # Subscribers:
        self._img_sub =self.create_subscription(Image, "/camera/camera/color/image_raw", self.image_callback, qos_profile_sensor_data)

        # Publishers:
        self._detections_pub = self.create_publisher(Detection2DArray, "detections", 10)
        self._annotated_img_pub = self.create_publisher(Image, "annotated_image", 10)

        # Logging some info
        self.get_logger().info(f"Subscribing to: {self._img_sub.topic_name}")
        self.get_logger().info(f"Publishing detections to: {self._detections_pub.topic_name}")
    
    #end of __init__


    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            #image[y1:y2, x1:x2]
            cropped_img = image[CROP_Y_OFFSET:720, CROP_X_OFFSET:1090] 

            # Run YOLO model on the converted image
            result = self._model(cropped_img, verbose=False)[0]

            # Create Detection2DArray msg
            detection_array = Detection2DArray()
            detection_array.header = msg.header

            # Process each detection
            for box in result.boxes:
                detection = Detection2D()
                detection.header = msg.header

                # Get bounding box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # # Set bounding box center and size (vision_msgs BoundingBox2D uses Pose2D for center (nested!))
                detection.bbox.center.position.x = float((x1 + x2) / 2) + CROP_X_OFFSET
                detection.bbox.center.position.y = float((y1 + y2) / 2) + CROP_Y_OFFSET

                detection.bbox.center.theta = 0.0 # no rotation for 2D detection
                detection.bbox.size_x = float(x2 - x1)
                detection.bbox.size_y = float(y2 - y1)

                # Draw bounding box onto cropped image
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                score = float(box.conf[0])

                text = f"{class_name}: {score:.2f}"
                color = PALETTE[class_id % 80].tolist()

                # cropped image
                # x1 = int(x1)
                # y1 = int(y1)
                # x2 = int(x2)
                # y2 = int(y2)
                # cv2.rectangle(cropped_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(cropped_img,text,(x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,)

                # original image: bounding box and class label with confidence score
                full_x1 = int(x1 + CROP_X_OFFSET)
                full_y1 = int(y1 + CROP_Y_OFFSET)
                full_x2 = int(x2 + CROP_X_OFFSET)
                full_y2 = int(y2 + CROP_Y_OFFSET)
                cv2.rectangle(image, (full_x1, full_y1), (full_x2, full_y2), color, 2)
                cv2.putText(image, text,(full_x1, max(full_y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,)

                # Set detection hypothesis (class and confidence)(vision_msgs uses ObjectHypothesisWithPose[] results)
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(class_id)
                hypothesis.hypothesis.score = score

                detection.results.append(hypothesis)
                detection_array.detections.append(detection)

            # adding a rectangle to show the cropped area on the original image
            cv2.rectangle(image, (CROP_X_OFFSET, CROP_Y_OFFSET), (1090, 720), (0, 0, 255), 3,)
            
            # Publish detections and annotated image
            self._detections_pub.publish(detection_array)

            #annotated_img = self._bridge.cv2_to_imgmsg(cropped_img, encoding="bgr8")
            annotated_img = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
            annotated_img.header = msg.header

            self._annotated_img_pub.publish(annotated_img)

            #self._annotated_img_pub.publish(self._bridge.cv2_to_imgmsg(result.plot(show=False), encoding="bgr8"))

        
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
    
    #end_of_publish_detections



def main(args=None):
    try:
        with rclpy.init(args=args):
            object_detector = ObjectDetector()

            rclpy.spin(object_detector)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()