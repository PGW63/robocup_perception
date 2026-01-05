import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import cv2
from ultralytics import YOLO

class Yolov8Node(Node):
    def __init__(self):
        super().__init__('yolov8_node')

        # Load the pre-trained YOLOv8 model (make sure to replace the path with your model's path)
        self.model = YOLO("yolov8s-oiv7.pt")  # Path to the YOLOv8 model you trained

        # Initialize CvBridge to convert ROS Image to OpenCV image
        self.bridge = CvBridge()

        # Create a subscriber to the image topic
        self.create_subscription(
            Image,
            'camera/camera/color/image_raw',  # Topic name (adjust to your camera topic)
            self.image_callback,
            10
        )

        self.pub_image = self.create_publisher(Image, 'yolo_oiv', 10)

        self.get_logger().info("YOLOv8 ROS2 Node has been started.")

    def image_callback(self, msg):
        try:
            # Convert the ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Perform detection with YOLOv8
            results = self.model(cv_image)

            result = results[0]

            # Use results.plot() to display the image with detections
            plots = result.plot()  # This method will show the image with bounding boxes, class labels, and scores

            ros_image = self.bridge.cv2_to_imgmsg(plots, 'bgr8')
            self.pub_image.publish(ros_image)


        except Exception as e:
            self.get_logger().error(f"Error in processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = Yolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
