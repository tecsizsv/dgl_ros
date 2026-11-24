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
                ('conf', 0.6),
                ('device', 'cuda:0'),
                ('use_yolo', False),
                ('cropping', False),
                ('crop', [440, 20, 1090, 720]),
                ('use_limit', False),
                ('limits', [0.25, 0.75, 0.25, 0.75]),
                ('file_logger', False),
                ('file_logger_dir_path', '/workspaces/vision-based-rgp-devcontainer/src/dgl_ros/dgl_ros_pipeline/logfiles'),
            ]
        )

        # Read parameter values
        model_name = self.get_parameter('model').value
        self.conf = self.get_parameter('conf').value
        self.device = self.get_parameter('device').value
        self.use_yolo = self.get_parameter('use_yolo').value
        self.cropping = self.get_parameter('cropping').value
        self.crop_x1, self.crop_y1, self.crop_x2, self.crop_y2 = self.get_parameter('crop').value
        self.use_limit = self.get_parameter('use_limit').value
        self.x_limit_low, self.x_limit_high, self.y_limit_low,  self.y_limit_high = self.get_parameter('limits').value
        self.file_logger = self.get_parameter('file_logger').value
        log_dir = Path(self.get_parameter('file_logger_dir_path').value)

        # Initialize YOLO model
        if self.use_yolo:
            self.get_logger().info(f'Loading YOLO model: {model_name}')
            self._model = YOLO(model_name)

            if '-world' in model_name:
                self._model.set_classes(self.get_parameter('classes').value)
        
        # Initialize subscribers and publishers
        self._img_sub =self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.image_callback, qos_profile_sensor_data)

        self._detections_pub = self.create_publisher(
            Detection2DArray, "detections", 10)
        self._annotated_img_pub = self.create_publisher(
            Image, "annotated_image", 10)

        # Configure file logger
        if self.file_logger and self.use_yolo:
            log_dir.mkdir(parents=True, exist_ok=True)
            yolo = Path(self._model.model_name).stem
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            self._file_logger = logging.getLogger(f'{self.get_name()}_file')
            self._file_logger.setLevel(logging.INFO)
            self._file_logger.propagate = False
            handler = logging.FileHandler(log_dir / f'object_detector_{yolo}_{stamp}.log', encoding='utf-8')
            handler.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d,%(message)s', '%Y-%m-%d %H:%M:%S'))
            self._file_logger.addHandler(handler)

        # Log node configuration
        self.get_logger().info(
            f'Object detector initialized :) \n'
            f'  YOLO enabled:       {self.use_yolo}\n'
            f'  Model:              {model_name}\n'
            f'  Device:             {self.device}\n'
            f'  Confidence:         {self.conf}\n'
            f'  Cropping enabled:   {self.cropping}\n'
            f'  Position filtering: {self.use_limit}\n'
            f'  File logging:       {self.file_logger and self.use_yolo}\n'
            f'  Input image topic:  {self._img_sub.topic_name}\n'
            f'  Detections topic:   {self._detections_pub.topic_name}\n'
            f'  Annotated topic:    {self._annotated_img_pub.topic_name}'
        )
    #end of __init__

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
                if self.cropping:
                    #image[y1:y2, x1:x2]
                    yolo_input = image[self.crop_y1:self.crop_y2, self.crop_x1:self.crop_x2]
                else:
                    yolo_input = image


                # Run YOLO
                result = self._model(yolo_input, verbose=False, conf=self.conf, device=self.device)[0]

                # Process and filter detections
                for box in result.boxes:
                    # Get bounding box coordinates (xyxy format)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    if self.cropping:
                        # Adjust bounding box coordinates to the original image
                        x1 += self.crop_x1
                        y1 += self.crop_y1
                        x2 += self.crop_x1
                        y2 += self.crop_y1

                    # Calculate bounding box center
                    cx = float((x1 + x2) / 2)
                    cy = float((y1 + y2) / 2)

                    # Reject detections outside the allowed center region
                    if self.use_limit and not (
                        self.x_limit_low * width <= cx <= self.x_limit_high * width 
                        and self.y_limit_low * height <= cy <= self.y_limit_high * height):
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
                    text = f"{class_name}: {score:.2f}"
                    color = self._colors(class_id, True)
                    annotator.box_label((x1, y1, x2, y2), text, color)

                    # Set object hypothesis with class ID and confidence score
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(class_id)
                    hypothesis.hypothesis.score = score

                    detection.results.append(hypothesis)
                    detection_array.detections.append(detection)

                

                # Log YOLO processing time [ms]
                if self.file_logger:
                    self._file_logger.info(sum(result.speed.values()))
                
            # Publish detection
            self._detections_pub.publish(detection_array)

            # Generate annotated image
            annotated_image = annotator.result()

            # Visualize crop region
            if self.use_yolo and self.cropping:
                cv2.rectangle(annotated_image, (self.crop_x1, self.crop_y1), (self.crop_x2, self.crop_y2), (0, 0, 255), 3,)

            # Publish annotated image
            annotated_img = self._bridge.cv2_to_imgmsg(annotated_image, encoding="bgr8")
            annotated_img.header = msg.header
            self._annotated_img_pub.publish(annotated_img)
            
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