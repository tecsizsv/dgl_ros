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

        self._colors = Colors()  # Use consistent colors for class IDs

        # Declare ROS 2 parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('model', 'yolo26m-objv1-150.pt'),
                ('classes', ['cardboard box', 'battery', 'bottle', 'rod', 'mug']),
                ('confidence_threshold', 0.6),
                ('device', 'cuda:0'),
                ('use_yolo', False),
                ('use_crop', False),
                ('crop_bounds', [440, 20, 1090, 720]),
                ('filter_by_position', False),
                ('position_limits', [0.25, 0.75, 0.25, 0.75]),
                ('enable_file_logging', False),
                ('log_dir', '/workspaces/vision-based-rgp-devcontainer/src/dgl_ros/dgl_ros_pipeline/logfiles'),
            ]
        )

        # Read parameter values
        model_name = self.get_parameter('model').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.device = self.get_parameter('device').value
        self.use_yolo = self.get_parameter('use_yolo').value
        self.use_crop = self.get_parameter('use_crop').value
        self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2 = self.get_parameter('crop_bounds').value
        self.filter_by_position = self.get_parameter('filter_by_position').value
        self.x_limit_min, self.x_limit_max, self.y_limit_min,  self.y_limit_max = self.get_parameter('position_limits').value
        self.enable_file_logging = self.get_parameter('enable_file_logging').value
        log_dir = Path(self.get_parameter('log_dir').value)/ 'object_detector'

        # Initialize YOLO model
        if self.use_yolo:
            self.get_logger().info(f'Loading YOLO model: {model_name}')
            self._model = YOLO(model_name)

            if '-world' in model_name:
                self._model.set_classes(self.get_parameter('classes').value)
        
        # Initialize subscribers and publishers
        self._image_sub =self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.image_callback, qos_profile_sensor_data)

        self._detections_pub = self.create_publisher(
            Detection2DArray, "detections", 10)
        self._annotated_image_pub = self.create_publisher(
            Image, "annotated_image", 10)

        # Configure file logger
        if self.enable_file_logging and self.use_yolo:
             self._setup_file_logger(log_dir)

        # Log node configuration
        self.get_logger().info(
            f'Object detector initialized :) \n'
            f'  YOLO enabled:       {self.use_yolo}\n'
            f'  Model:              {model_name}\n'
            f'  Device:             {self.device}\n'
            f'  Confidence:         {self.confidence_threshold}\n'
            f'  Cropping enabled:   {self.use_crop}\n'
            f'  Position filtering: {self.filter_by_position}\n'
            f'  File logging:       {self.enable_file_logging and self.use_yolo}\n'
            f'  Input image topic:  {self._image_sub.topic_name}\n'
            f'  Detections topic:   {self._detections_pub.topic_name}\n'
            f'  Annotated topic:    {self._annotated_image_pub.topic_name}'
        )

    def _setup_file_logger(self, log_dir):
        """Configure the file logger for YOLO processing times."""
        log_dir.mkdir(parents=True, exist_ok=True)
        model_name = Path(self._model.model_name).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        self._file_logger = logging.getLogger(f'{self.get_name()}_file')
        self._file_logger.setLevel(logging.INFO)
        self._file_logger.propagate = False
        handler = logging.FileHandler(log_dir / f'object_detector_{model_name}_{timestamp}.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d,%(message)s', '%Y-%m-%d %H:%M:%S'))
        self._file_logger.addHandler(handler)

    def image_callback(self, msg):
        """Run detection on the incoming image and publish the detections and the annotated image."""

        try:
            # Convert ROS Image to OpenCV format
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            width = image.shape[1]
            height = image.shape[0]

            # Initialize detection array and image annotator
            detection_array = Detection2DArray()
            detection_array.header = msg.header
            annotator = Annotator(image, line_width=3)

            # Generate a fixed central bounding box when YOLO is disabled    
            if not self.use_yolo:
                detection = Detection2D()
                detection.header = msg.header
                detection.bbox.center.position.x = width / 2.0
                detection.bbox.center.position.y = height / 2.0
                detection.bbox.center.theta = 0.0
                detection.bbox.size_x = width * 0.5
                detection.bbox.size_y = height * 0.5

                detection_array.detections.append(detection)

                # Annotate the fixed bounding box
                color = self._colors(0, True)
                annotator.box_label((width * 0.25, height * 0.25, width * 0.75, height * 0.75), "", color)
            else:
                # Prepare YOLO input
                if self.use_crop:
                    #image[y1:y2, x1:x2]
                    yolo_input = image[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]
                else:
                    yolo_input = image

                # Run YOLO
                result = self._model(yolo_input, verbose=False, conf=self.confidence_threshold, device=self.device)[0]

                # Process and filter detections
                for box in result.boxes:
                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    if self.use_crop:
                        # Adjust bounding box coordinates to the original image
                        x1 += self.crop_x1
                        y1 += self.crop_y1
                        x2 += self.crop_x1
                        y2 += self.crop_y1

                    # Calculate bounding box center
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)

                    # Reject detections outside the allowed center region
                    if self.filter_by_position and not (
                        self.x_limit_min * width <= cx <= self.x_limit_max * width 
                        and self.y_limit_min * height <= cy <= self.y_limit_max * height
                    ):
                        continue

                    # Create detection message 
                    detection = Detection2D()
                    detection.header = msg.header
                    detection.bbox.center.theta = 0.0 # Axis-aligned bounding box
                    detection.bbox.center.position.x = cx
                    detection.bbox.center.position.y = cy
                    detection.bbox.size_x = float(x2 - x1)
                    detection.bbox.size_y = float(y2 - y1)

                    # Get class ID, class name, and confidence score
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    score = float(box.conf[0])

                    # Annotate the image with bounding boxes and class labels with confidence score
                    label = f"{class_name}: {score:.2f}"
                    color = self._colors(class_id, True)
                    annotator.box_label((x1, y1, x2, y2), label, color)

                    # Set object hypothesis with class ID and confidence score
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(class_id)
                    hypothesis.hypothesis.score = score

                    detection.results.append(hypothesis)
                    detection_array.detections.append(detection)  

                # Log YOLO processing time [ms]
                if self.enable_file_logging:
                    self._file_logger.info(sum(result.speed.values()))
                
            # Publish detections
            self._detections_pub.publish(detection_array)

            # Generate annotated image
            annotated_image = annotator.result()

            # Visualize crop region
            if self.use_yolo and self.use_crop:
                cv2.rectangle(annotated_image, (self.crop_x1, self.crop_y1), (self.crop_x2, self.crop_y2), (0, 0, 255), 3,)

            # Publish annotated image
            annotated_image_msg = self._bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
            annotated_image_msg.header = msg.header
            self._annotated_image_pub.publish(annotated_image_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    try:
        with rclpy.init(args=args):
            object_detector = ObjectDetector()

            rclpy.spin(object_detector)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass


if __name__ == '__main__':
    main()