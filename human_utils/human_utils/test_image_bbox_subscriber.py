#!/usr/bin/env python3
"""
Test ImageBbox Subscriber Node
- inha_interfaces/ImageBbox 메시지를 구독해서 각 필드 출력
- 메시지가 제대로 전달되는지 확인용
"""

import rclpy
from rclpy.node import Node
from inha_interfaces.msg import ImageBbox
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class TestImageBboxSubscriber(Node):
    def __init__(self):
        super().__init__('test_image_bbox_subscriber')
        
        # 파라미터
        self.declare_parameter("topic", "/human/hand_up_image_bbox")
        topic = self.get_parameter("topic").value
        
        # QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        
        # 구독자
        self.sub = self.create_subscription(
            ImageBbox, topic, self.callback, qos
        )
        
        self.msg_count = 0
        
        self.get_logger().info(f"Test subscriber started, listening to: {topic}")
    
    def callback(self, msg: ImageBbox):
        """메시지 수신 시 각 필드 출력"""
        self.msg_count += 1
        
        self.get_logger().info("=" * 60)
        self.get_logger().info(f"📩 Message #{self.msg_count} received!")
        self.get_logger().info("=" * 60)
        
        # BoundingBox2D 정보
        self.get_logger().info("📦 BoundingBox2D (inha_bbox):")
        self.get_logger().info(f"  - Center X: {msg.inha_bbox.center.position.x}")
        self.get_logger().info(f"  - Center Y: {msg.inha_bbox.center.position.y}")
        self.get_logger().info(f"  - Theta: {msg.inha_bbox.center.theta}")
        self.get_logger().info(f"  - Size X (Width): {msg.inha_bbox.size_x}")
        self.get_logger().info(f"  - Size Y (Height): {msg.inha_bbox.size_y}")
        
        # 이미지 정보
        self.get_logger().info("🖼️  CompressedImage (inha_image):")
        self.get_logger().info(f"  - Format: {msg.inha_image.format}")
        self.get_logger().info(f"  - Data size: {len(msg.inha_image.data)} bytes")
        self.get_logger().info(f"  - Timestamp: {msg.inha_image.header.stamp.sec}.{msg.inha_image.header.stamp.nanosec}")
        self.get_logger().info(f"  - Frame ID: {msg.inha_image.header.frame_id}")
        
        self.get_logger().info("=" * 60)


def main(args=None):
    rclpy.init(args=args)
    node = TestImageBboxSubscriber()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
