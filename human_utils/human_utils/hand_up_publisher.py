#!/usr/bin/env python3
"""
Hand Up Publisher Node
- /human/hand_up_detected (Bool) 토픽을 구독
- hand up이 감지되면 inha_interfaces/ImageBbox 메시지 발행
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray
from sensor_msgs.msg import CompressedImage
from vision_msgs.msg import BoundingBox2D
from inha_interfaces.msg import ImageBbox
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cv2


class HandUpPublisher(Node):
    def __init__(self):
        super().__init__('hand_up_publisher_node')
        
        # 파라미터 선언
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw/compressed")
        self.declare_parameter("output_topic", "/human/hand_up_image_bbox")
        self.declare_parameter("continuous_publish", True) # True: 계속 발행, False: 한 번만
        self.declare_parameter("publish_rate", 2.0)  # 계속 발행 시 주기 (Hz)
        
        # 파라미터 로드
        self.image_topic = self.get_parameter("image_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.continuous_publish = self.get_parameter("continuous_publish").value
        self.publish_rate = self.get_parameter("publish_rate").value
        
        # QoS 설정
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        
        # 상태 저장
        self.latest_image = None
        self.latest_bbox = None  # [x1, y1, x2, y2]
        self.hand_up_triggered = False
        self.hand_up_active = False  # hand up이 활성화되어 있는지
        
        # 타이머 (계속 발행 모드용)
        if self.continuous_publish:
            self.publish_timer = self.create_timer(1.0 / self.publish_rate, self.publish_callback)
        else:
            self.publish_timer = None
        
        # 구독자
        self.sub_hand_up = self.create_subscription(
            Bool, "/human/hand_up_detected", self.on_hand_up, 10
        )
        self.sub_bbox = self.create_subscription(
            Float32MultiArray, "/human/hand_up_bbox", self.on_bbox, 10
        )
        self.sub_image = self.create_subscription(
            CompressedImage, self.image_topic, self.on_image, qos_best_effort
        )
        
        # 발행자
        self.pub_image_bbox = self.create_publisher(ImageBbox, self.output_topic, qos_best_effort)
        
        self.get_logger().info("Hand Up Publisher Node initialized")
        self.get_logger().info(f"  Subscribing to: /human/hand_up_detected")
        self.get_logger().info(f"  Publishing to: {self.output_topic}")
        self.get_logger().info(f"  Continuous publish: {self.continuous_publish}")
        if self.continuous_publish:
            self.get_logger().info(f"  Publish rate: {self.publish_rate} Hz")
    
    def on_image(self, msg: CompressedImage):
        """이미지 저장"""
        self.latest_image = msg
    
    def on_bbox(self, msg: Float32MultiArray):
        """Bbox 정보 저장 [x1, y1, x2, y2]"""
        if len(msg.data) == 4:
            self.latest_bbox = msg.data
    
    def on_hand_up(self, msg: Bool):
        """Hand up 감지 시 처리"""
        if msg.data:
            # Hand up 감지됨
            if not self.hand_up_active:
                self.hand_up_active = True
                self.get_logger().info("✋ Hand up activated!")
                
                # 한 번만 발행 모드면 즉시 발행
                if not self.continuous_publish:
                    self.publish_image_bbox()
        else:
            # Hand up 해제됨
            if self.hand_up_active:
                self.hand_up_active = False
                self.hand_up_triggered = False
                self.get_logger().info("Hand up deactivated")
    
    def publish_callback(self):
        """타이머 콜백 (계속 발행 모드)"""
        if self.hand_up_active:
            self.publish_image_bbox()
    
    def publish_image_bbox(self):
        """ImageBbox 메시지 발행"""
        # 한 번만 발행 모드에서 이미 발행했으면 무시
        if not self.continuous_publish and self.hand_up_triggered:
            return
        
        self.hand_up_triggered = True
        
        if self.latest_image is None:
            self.get_logger().warn("No image available to publish")
            return
        
        if self.latest_bbox is None:
            self.get_logger().warn("No bbox available to publish")
            return
        
        # ImageBbox 메시지 생성
        image_bbox_msg = ImageBbox()
        
        # 이미지 설정
        image_bbox_msg.inha_image = self.latest_image
        
        # BoundingBox2D 설정 (hand up한 사람의 bbox)
        try:
            x1, y1, x2, y2 = self.latest_bbox
            
            # 중심점과 크기 계산
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1
            
            bbox = BoundingBox2D()
            bbox.center.position.x = float(center_x)
            bbox.center.position.y = float(center_y)
            bbox.center.theta = 0.0
            bbox.size_x = float(width)
            bbox.size_y = float(height)
            
            image_bbox_msg.inha_bbox = bbox
            
            # 메시지 발행
            self.pub_image_bbox.publish(image_bbox_msg)
            self.get_logger().info(f"✅ Published ImageBbox message (bbox: {bbox.size_x:.1f}x{bbox.size_y:.1f})")
            
        except Exception as e:
            self.get_logger().error(f"Failed to create ImageBbox message: {e}")
    
    def reset_trigger(self):
        """트리거 리셋 (외부에서 호출 가능)"""
        self.hand_up_triggered = False
        self.get_logger().info("Hand up trigger reset")


def main(args=None):
    rclpy.init(args=args)
    node = HandUpPublisher()
    
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
