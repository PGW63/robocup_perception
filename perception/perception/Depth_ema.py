#!/home/nvidia/vision_ws/src/detection_venv/bin/python
"""
Depth EMA Filter Node

기능:
- Aligned depth 이미지를 구독
- EMA (Exponential Moving Average) 필터 적용
- 필터링된 depth 이미지 퍼블리시
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
import numpy as np


class DepthEMANode(Node):
    def __init__(self):
        super().__init__('depth_ema_node')
        
        # Parameters
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('alpha', 0.3)  # EMA smoothing factor (0~1)
        
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.alpha = self.get_parameter('alpha').get_parameter_value().double_value
        
        # self.bridge = CvBridge()
        
        # EMA state
        self.depth_ema = None
        
        # Subscriber
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10
        )
        
        # Publisher
        self.filtered_depth_pub = self.create_publisher(
            Image,
            '/detection_node/filtered_depth_image',
            10
        )
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("Depth EMA Filter Node Started")
        self.get_logger().info(f"  - Input topic: {depth_topic}")
        self.get_logger().info(f"  - Output topic: /detection_node/filtered_depth")
        self.get_logger().info(f"  - EMA alpha: {self.alpha}")
        self.get_logger().info("=" * 60)
    
    def depth_callback(self, msg):
        """Depth 이미지 콜백 - EMA 필터 적용"""
        # self.get_logger().info("Get Depth Image")
        try:
            # Convert ROS Image to numpy array
            # depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            # self.get_logger().info(f"Received depth image: {msg.width}x{msg.height}, encoding={msg.encoding}, step={msg.step}")
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
            # self.get_logger().info(f"Depth array shape: {depth.shape}, min={depth.min()}, max={depth.max()}")
            # Apply EMA filter
            depth_filtered = self.apply_ema(depth)
            # self.get_logger().info(f"Filtered depth: min={depth_filtered.min()}, max={depth_filtered.max()}")
            
            # Convert back to ROS Image message
            # filtered_msg = self.bridge.cv2_to_imgmsg(
            #     depth_filtered.astype(depth.dtype), 
            #     encoding='passthrough'
            # )
            filtered_msg = Image()
            filtered_msg.header = msg.header
            filtered_msg.height = msg.height
            filtered_msg.width = msg.width
            filtered_msg.encoding = msg.encoding
            filtered_msg.is_bigendian = msg.is_bigendian
            filtered_msg.step = msg.step
            filtered_msg.data = depth_filtered.astype(np.uint16).tobytes()
            # self.get_logger().debug("Published filtered depth image")
            # Publish filtered depth image
            self.filtered_depth_pub.publish(filtered_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {e}")
    
    def apply_ema(self, depth_new):
        """
        EMA (Exponential Moving Average) 필터 적용
        
        Formula: EMA[t] = alpha * depth[t] + (1 - alpha) * EMA[t-1]
        - alpha가 클수록 새로운 값에 민감 (빠른 반응)
        - alpha가 작을수록 이전 값을 더 유지 (부드러운 변화)
        """
        # self.get_logger().info("Apply EMA")
        if self.depth_ema is None:
            # 첫 프레임은 그대로 사용
            self.depth_ema = depth_new.copy().astype(np.float32)
        else:
            # 유효한 depth 값만 업데이트 (0보다 큰 값)
            valid_mask = depth_new > 0
            depth_float = depth_new.astype(np.float32)
            
            # EMA 업데이트
            self.depth_ema[valid_mask] = (
                self.alpha * depth_float[valid_mask] + 
                (1 - self.alpha) * self.depth_ema[valid_mask]
            )
        
        return self.depth_ema


def main(args=None):
    rclpy.init(args=args)
    node = DepthEMANode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()