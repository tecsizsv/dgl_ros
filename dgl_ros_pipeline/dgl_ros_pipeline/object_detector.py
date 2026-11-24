import cv2
import rclpy
import cv_bridge
import logging
from datetime import datetime
from pathlib import Path
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, Colors
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from rclpy.qos import qos_profile_sensor_data # qos_profile_sensor_data: Quality of Service (QoS) profile

class ObjectDetector(Node):
    """Detect objects with YOLO on the color image and publish them as Detection2DArray."""

    def __init__(self):
        """Load the YOLO model and set up the parameters, subscriber, publishers and file logger."""

        super().__init__('object_detector')
        self._bridge = cv_bridge.CvBridge()

        self._colors = Colors()  # For consistent colors across runs
        
        # Load the YOLO model:
        self.declare_parameter('model', 'yolo26m-objv1-150.pt')
        model_name = self.get_parameter('model').value
        self._model = YOLO(model_name)

        # Class prompts, only used with a YOLO-World model
        self.declare_parameter('classes', ["cardboard box", "battery", "bottle", "rod","mug"])
        if '-world' in model_name:
            self._model.set_classes(self.get_parameter('classes').value)

        # Detection confidence threshold
        self.declare_parameter('conf', 0.6)
        self.conf = self.get_parameter('conf').value

        # Device to run YOLO on
        self.declare_parameter('device', "cuda:0")
        self.device = self.get_parameter('device').value

        # Crop area in pixels: x1, y1, x2, y2
        self.declare_parameter('crop', [440, 20, 1090, 720])
        self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2 = self.get_parameter('crop').value

        # Enable image cropping
        self.declare_parameter('cropping', False)
        self.cropping = self.get_parameter('cropping').value
        self.get_logger().info(f'Using cropped image: {self.cropping}')

        # Allowed bbox center area as fraction of image size: x_min, x_max, y_min, y_max
        self.declare_parameter('limits', [0.25, 0.75, 0.25, 0.75])
        self.x_limit_low, self.x_limit_high, self.y_limit_low,  self.y_limit_high = self.get_parameter('limits').value

        # Enable bounding box position filtering
        self.declare_parameter('use_limit', False)
        self.use_limit = self.get_parameter('use_limit').value
        self.get_logger().info(f'Filtering bounding box position: {self.use_limit}')
    
        # Enable YOLO-based detection
        self.declare_parameter('use_yolo', False)
        self.use_yolo = self.get_parameter('use_yolo').value
        self.get_logger().info(f'Using YOLO: {self.use_yolo}')

        # Subscribers:
        self._img_sub =self.create_subscription(Image, "/camera/camera/color/image_raw", self.image_callback, qos_profile_sensor_data)
        self.get_logger().info(f"Subscribing to: {self._img_sub.topic_name}")

        # Publishers:
        self._detections_pub = self.create_publisher(Detection2DArray, "detections", 10)
        self._annotated_img_pub = self.create_publisher(Image, "annotated_image", 10)
        self.get_logger().info(f"Publishing detections to: {self._detections_pub.topic_name}")

        # File logger setup
        self.declare_parameter('file_logger', False)
        self.declare_parameter('file_logger_dir_path', '/workspaces/vision-based-rgp-devcontainer/src/dgl_ros/dgl_ros_pipeline/logfiles')
        self.file_logger = self.get_parameter('file_logger').value

        if self.file_logger:
            log_dir = Path( self.get_parameter('file_logger_dir_path').value)
            yolo = Path(self._model.model_name).stem
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            self._file_logger = logging.getLogger(f'{self.get_name()}_file')
            self._file_logger.setLevel(logging.INFO)
            self._file_logger.propagate = False
            handler = logging.FileHandler(log_dir / f'object_detector_{yolo}_{stamp}.log', encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d,%(message)s', '%Y-%m-%d %H:%M:%S'))
            self._file_logger.addHandler(handler)
    #end of __init__

    def image_callback(self, msg):
        """Run detection on the incoming image and publish the detections and the annotated image."""

        try:
            # Convert ROS Image to OpenCV format
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            if self.cropping:
                #image[y1:y2, x1:x2]
                yolo_input = image[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]
            else:
                yolo_input = image

            width = image.shape[1]
            height = image.shape[0]

            # Generate a fixed central bounding box when YOLO is disabled    
            if not self.use_yolo:
                detection = Detection2D()
                detection.header = msg.header
                detection.bbox.center.position.x = width / 2.0
                detection.bbox.center.position.y = height / 2.0
                detection.bbox.center.theta = 0.0
                detection.bbox.size_x = width * 0.5
                detection.bbox.size_y = height * 0.5
                detection_array = Detection2DArray()
                detection_array.header = msg.header
                detection_array.detections.append(detection)
                self._detections_pub.publish(detection_array)

                # Annotate the image with bounding boxes and class labels with confidence score
                annotator = Annotator(image, line_width=3)
                color = self._colors(0, True)
                annotator.box_label((width / 2.0 - detection.bbox.size_x / 2.0, height / 2.0 - detection.bbox.size_y / 2.0, width / 2.0 + detection.bbox.size_x / 2.0, height / 2.0 + detection.bbox.size_y / 2.0), "", color)
                
                image2 = annotator.result()
                annotated_img = self._bridge.cv2_to_imgmsg(image2, encoding="bgr8")
                annotated_img.header = msg.header
                self._annotated_img_pub.publish(annotated_img)
                return
            else:
                # Run YOLO model on the converted image
                result = self._model(yolo_input, verbose=False, conf=self.conf, device=self.device)[0]

                # Create Detection2DArray msg
                detection_array = Detection2DArray()
                detection_array.header = msg.header

                annotator = Annotator(image, line_width=3)

                # Process each detection
                for box in result.boxes:
                    detection = Detection2D()
                    detection.header = msg.header

                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    if self.cropping:
                        # Adjust bounding box coordinates to the original image if cropped image used
                        x1 += self.crop_x1
                        y1 += self.crop_y1
                        x2 += self.crop_x1
                        y2 += self.crop_y1

                    # Set bounding box center and size (vision_msgs BoundingBox2D uses Pose2D for center (nested!))
                    detection.bbox.center.position.x = float((x1 + x2) / 2)
                    detection.bbox.center.position.y = float((y1 + y2) / 2)


                    if self.use_limit:
                        if not (self.x_limit_low * width <= detection.bbox.center.position.x <= self.x_limit_high * width and self.y_limit_low * height <= detection.bbox.center.position.y <= self.y_limit_high * height):
                            continue

                    detection.bbox.center.theta = 0.0 # Axis-aligned bounding box (no rotation for 2D objects)
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)


                    # Get class ID, class name, and confidence score
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    score = float(box.conf[0])

                    # Annotate the image with bounding boxes and class labels with confidence score
                    text = f"{class_name}: {score:.2f}"
                    color = self._colors(class_id, True)
                    annotator.box_label((x1, y1, x2, y2), text, color)

                    # Set detection hypothesis (class and confidence)(vision_msgs uses ObjectHypothesisWithPose[] results)
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(class_id)
                    hypothesis.hypothesis.score = score

                    detection.results.append(hypothesis)
                    detection_array.detections.append(detection)

                image = annotator.result()
                # Drawing a rectangle to show the cropped area on the original image
                if self.cropping:
                    cv2.rectangle(image, (self.crop_x1, self.crop_y1), (self.crop_x2, self.crop_y2), (0, 0, 255), 3,)
                
                # Publish detections and annotated image
                self._detections_pub.publish(detection_array)

                annotated_img = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
                annotated_img.header = msg.header
                self._annotated_img_pub.publish(annotated_img)

                # Log total YOLO processing time in milliseconds
                if self.file_logger:
                    self._file_logger.info(sum(result.speed.values()))
        
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
    #end of image_callback

def main(args=None):
    try:
        with rclpy.init(args=args):
            object_detector = ObjectDetector()

            rclpy.spin(object_detector)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()