#!/home/nvidia/vision_ws/src/detection_venv/bin/python
"""
YOLO Detection Node for ROS2 Humble
- 모든 객체를 최대한 빠르게 탐지
- /detection_node/use_open_set 토픽으로 GroundedSAM2와 연동 제어
- use_open_set=False일 때만 동작 (기본 모드)
- vision_msgs/Detection2DArray로 결과 퍼블리시
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Bool
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from rcl_interfaces.msg import SetParametersResult
from ament_index_python.packages import get_package_share_directory
import os


class YoloNode(Node):
    def __init__(self):
        super().__init__('yolo_node')

        package_name = "perception"
        package_dir = get_package_share_directory(package_name)

        # Parameters
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('image_type', 'compressed')  # 'raw' or 'compressed'
        self.declare_parameter('set_segmentation', False)
        self.declare_parameter('confidence_threshold', 0.5)

        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.image_type = self.get_parameter('image_type').get_parameter_value().string_value
        self.segmentation = bool(self.get_parameter('set_segmentation').get_parameter_value().bool_value)
        self.confidence_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        
        # use_open_set=False면 YOLO 활성화, True면 비활성화 (GroundedSAM2가 동작)
        self.use_open_set = False

        self.bridge = CvBridge()

        # Publishers (공통 토픽)
        self.debug_image_pub = self.create_publisher(Image, '/detection_node/debug_image', 10)
        self.detection_pub = self.create_publisher(Detection2DArray, '/detection_node/detections', 10)

        # Subscriber for mode control (GroundedSAM2와 공유)
        self.mode_sub = self.create_subscription(
            Bool,
            '/detection_node/use_open_set',
            self.mode_callback,
            10)

        # Camera subscription (동적으로 생성)
        self.subscription = None
        self._create_image_subscription()

        # Load YOLO models
        self.model = YOLO(os.path.join(package_dir, 'models', 'yolov8s.pt'))
        self.segmentation_model = YOLO(os.path.join(package_dir, 'models', 'yolov8s-seg.pt'))

        self._cb_handle = self.add_on_set_parameters_callback(self.on_param_change)

        self.get_logger().info("=" * 50)
        self.get_logger().info(f"YOLO Node initialized")
        self.get_logger().info(f"  - Image topic: {self.image_topic} ({self.image_type})")
        self.get_logger().info(f"  - Segmentation: {self.segmentation}")
        self.get_logger().info(f"  - Detects: ALL objects (no filtering)")
        self.get_logger().info(f"  - Mode control: /detection_node/use_open_set")
        self.get_logger().info(f"      False -> YOLO active (default)")
        self.get_logger().info(f"      True  -> GroundedSAM2 active")
        self.get_logger().info(f"  - Output: /detection_node/detections, /detection_node/debug_image")
        self.get_logger().info("=" * 50)

    def _create_image_subscription(self):
        """Create or recreate image subscription based on image_type"""
        if self.subscription is not None:
            self.destroy_subscription(self.subscription)
        
        if self.image_type == 'raw':
            self.subscription = self.create_subscription(
                Image,
                self.image_topic,
                self.image_raw_callback,
                10)
        else:
            self.subscription = self.create_subscription(
                CompressedImage,
                self.image_topic,
                self.image_compressed_callback,
                10)

    def mode_callback(self, msg: Bool):
        """모드 전환: use_open_set=False면 YOLO, True면 GroundedSAM2"""
        self.use_open_set = msg.data
        if not self.use_open_set:
            self.get_logger().info("Mode: YOLO active")
        else:
            self.get_logger().info("Mode: GroundedSAM2 active (YOLO paused)")

    def on_param_change(self, params):
        for p in params:
            if p.name == 'set_segmentation':
                if p.type_ != p.Type.BOOL:
                    return SetParametersResult(successful=False, reason='set_segmentation must be bool')
                self.segmentation = bool(p.value)
                self.get_logger().info(f"segmentation -> {self.segmentation}")
            elif p.name == 'confidence_threshold':
                self.confidence_threshold = p.value
                self.get_logger().info(f"confidence_threshold -> {self.confidence_threshold}")
        
        return SetParametersResult(successful=True)

    
    def image_raw_callback(self, msg):
        if self.use_open_set:  # GroundedSAM2 모드면 YOLO는 동작 안함
            return
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.process(cv_image, msg.header)

    def image_compressed_callback(self, msg):
        if self.use_open_set:  # GroundedSAM2 모드면 YOLO는 동작 안함
            return
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        self.process(cv_image, msg.header)

    
    def process(self, img, header):
        img_copy = img.copy()
        detections_list = []  # Detection2D 리스트
        
        if self.segmentation:
            results = self.segmentation_model(img, verbose=False)
            res = results[0]  

            if res.boxes is not None and len(res.boxes) > 0:
                for i in range(len(res.boxes)):
                    c = int(res.boxes.cls[i].item())
                    conf = float(res.boxes.conf[i].item())
                    
                    if conf < self.confidence_threshold:
                        continue

                    x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy().astype(int)
                    label = res.names[c]

                    # Create Detection2D message
                    det = self._create_detection2d(x1, y1, x2, y2, label, conf, header)
                    detections_list.append(det)

                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_copy, f"{label} {conf:.2f}", (x1, max(0, y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if res.masks is not None and i < len(res.masks.data):
                        m = res.masks.data[i].cpu().numpy().astype(bool)
                        if m.shape == img_copy.shape[:2]:
                            img_copy[m] = (img_copy[m] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)

            self.publish_output(img_copy, detections_list, header)

        else:
            results = self.model(img, verbose=False)
            res = results[0]

            if res.boxes is not None and len(res.boxes) > 0:
                for i in range(len(res.boxes)):
                    c = int(res.boxes.cls[i].item())
                    conf = float(res.boxes.conf[i].item())
                    
                    if conf < self.confidence_threshold:
                        continue

                    x1, y1, x2, y2 = res.boxes.xyxy[i].cpu().numpy().astype(int)
                    label = res.names[c]

                    # Create Detection2D message
                    det = self._create_detection2d(x1, y1, x2, y2, label, conf, header)
                    detections_list.append(det)

                    cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img_copy, f"{label} {conf:.2f}", (x1, max(0, y1-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            self.publish_output(img_copy, detections_list, header)

    def _create_detection2d(self, x1, y1, x2, y2, label, confidence, header):
        """Create a Detection2D message from bounding box coordinates"""
        det = Detection2D()
        det.header = header
        
        # BBox center and size
        det.bbox.center.position.x = float((x1 + x2) / 2)
        det.bbox.center.position.y = float((y1 + y2) / 2)
        det.bbox.size_x = float(x2 - x1)
        det.bbox.size_y = float(y2 - y1)
        
        # Hypothesis with class id and score
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = label
        hyp.hypothesis.score = confidence
        det.results.append(hyp)
        
        return det

    def publish_output(self, img, detections_list, header):
        # Publish debug image
        debug_image = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        debug_image.header = header
        self.debug_image_pub.publish(debug_image)
        
        # Publish Detection2DArray
        det_array = Detection2DArray()
        det_array.header = header
        det_array.detections = detections_list
        self.detection_pub.publish(det_array)

def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
