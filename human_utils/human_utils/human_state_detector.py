#!/usr/bin/env python3
"""
Human State Detector Node
- utils.py의 기하학 함수 활용
- 서있기, 앉아있기, 누워있기, 손들기 판별
- 디버그 이미지에 사람별 상태 표시
"""

import numpy as np
import cv2
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, ColorRGBA, Bool
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
# from cv_bridge import CvBridge
from ultralytics import YOLO
import tf2_ros
from tf2_ros import TransformBroadcaster

# human_state_3d에서 가져오기
from human_utils.utils import (
    Point3D, Skeleton3D, KeypointIndex, SKELETON_CONNECTIONS,
    angle_3d, angle_3d_on_plane, distance_3d, vector_3d, midpoint_3d,
    JointAngleCalculator
)


# ============================================================
# Human State Enum
# ============================================================
class HumanState(Enum):
    UNKNOWN = auto()
    STANDING = auto()      # 서있기
    SITTING = auto()       # 앉아있기
    LYING_DOWN = auto()    # 누워있기
    HAND_UP_LEFT = auto()  # 왼손 들기
    HAND_UP_RIGHT = auto() # 오른손 들기
    HAND_UP_BOTH = auto()  # 양손 들기


# 상태별 색상 (BGR)
STATE_COLORS = {
    HumanState.UNKNOWN: (128, 128, 128),      # 회색
    HumanState.STANDING: (0, 255, 0),         # 초록
    HumanState.SITTING: (255, 165, 0),        # 주황
    HumanState.LYING_DOWN: (255, 0, 0),       # 파랑
    HumanState.HAND_UP_LEFT: (0, 255, 255),   # 노랑
    HumanState.HAND_UP_RIGHT: (0, 255, 255),  # 노랑
    HumanState.HAND_UP_BOTH: (0, 0, 255),     # 빨강
}


# ============================================================
# Rule-based 상태 판별기
# ============================================================
class HumanStateDetector:
    """Rule-based 기하학으로 사람 상태 판별"""
    
    def __init__(self):
        # === 손 들기 임계값 ===
        self.hand_up_wrist_above_shoulder = True  # 손목이 어깨보다 위
        self.hand_up_elbow_angle_min = 150.0       # 팔꿈치 최소 각도 (90도 이상이면 펴짐)
        
        # === 앉기 임계값 ===
        self.sitting_knee_angle_max = 130.0      # 무릎 각도 (130도 이하면 구부림)
        self.sitting_hip_angle_max = 130.0       # 엉덩이 각도 (130도 이하면 앉음)
        
        # === 누워있기 임계값 ===
        self.lying_torso_horizontal_threshold = 20.0  # 몸통 기울기 (45도 이하면 누움)
        
        # === 서있기 임계값 ===
        self.standing_knee_angle_min = 150.0     # 무릎 각도 (150도 이상이면 펴짐)
        self.standing_hip_angle_min = 150.0      # 엉덩이 각도 (150도 이상이면 펴짐)

    def detect_state(self, skeleton: Skeleton3D) -> Tuple[HumanState, Dict[str, float]]:
        """
        스켈레톤에서 상태 판별
        
        Returns:
            Tuple[HumanState, Dict]: (상태, 판별에 사용된 각도들)
        """
        angles = JointAngleCalculator.get_all_angles(skeleton)
        
        # 1. 손 들기 먼저 체크 (다른 상태와 병행 가능)
        hand_state = self._detect_hand_up(skeleton, angles)
        if hand_state:
            return hand_state, angles
        
        # 2. 누워있기 체크
        if self._is_lying_down(skeleton, angles):
            return HumanState.LYING_DOWN, angles
        
        # 3. 앉아있기 체크
        if self._is_sitting(skeleton, angles):
            return HumanState.SITTING, angles
        
        # 4. 서있기 체크
        if self._is_standing(skeleton, angles):
            return HumanState.STANDING, angles
        
        return HumanState.UNKNOWN, angles
    
    def _detect_hand_up(self, skel: Skeleton3D, angles: Dict) -> Optional[HumanState]:
        """손 들기 감지"""
        left_up = self._is_hand_up(skel, "left", angles)
        right_up = self._is_hand_up(skel, "right", angles)
        
        if left_up and right_up:
            return HumanState.HAND_UP_BOTH
        elif left_up:
            return HumanState.HAND_UP_LEFT
        elif right_up:
            return HumanState.HAND_UP_RIGHT
        return None
    
    def _is_hand_up(self, skel: Skeleton3D, side: str, angles: Dict) -> bool:
        """한 쪽 손이 들려있는지 확인"""
        if side == "left":
            shoulder_idx = KeypointIndex.LEFT_SHOULDER
            elbow_idx = KeypointIndex.LEFT_ELBOW
            wrist_idx = KeypointIndex.LEFT_WRIST
            elbow_angle_key = "left_elbow"
        else:
            shoulder_idx = KeypointIndex.RIGHT_SHOULDER
            elbow_idx = KeypointIndex.RIGHT_ELBOW
            wrist_idx = KeypointIndex.RIGHT_WRIST
            elbow_angle_key = "right_elbow"
        
        # 필요한 키포인트 확인
        if not all(skel.has_point(i) for i in [shoulder_idx, elbow_idx, wrist_idx]):
            return False
        
        shoulder = skel.keypoints[shoulder_idx]
        wrist = skel.keypoints[wrist_idx]
        
        # 조건 1: 손목이 어깨보다 위에 있는지
        # 카메라 좌표계: Y가 아래로 증가하므로, wrist.y < shoulder.y이면 위
        wrist_above_shoulder = wrist.y < shoulder.y
        
        # 조건 2: 팔꿈치 각도가 충분히 펴져있는지
        elbow_angle = angles.get(elbow_angle_key)
        elbow_extended = elbow_angle is not None and elbow_angle >= self.hand_up_elbow_angle_min
        
        return wrist_above_shoulder
    
    def _is_lying_down(self, skel: Skeleton3D, angles: Dict) -> bool:
        """누워있는 자세 감지"""
        # 어깨와 엉덩이가 필요
        if not (skel.has_point(KeypointIndex.LEFT_SHOULDER) and 
                skel.has_point(KeypointIndex.LEFT_HIP)):
            if not (skel.has_point(KeypointIndex.RIGHT_SHOULDER) and 
                    skel.has_point(KeypointIndex.RIGHT_HIP)):
                return False
        
        # 왼쪽 또는 오른쪽 어깨-엉덩이 사용
        if skel.has_point(KeypointIndex.LEFT_SHOULDER) and skel.has_point(KeypointIndex.LEFT_HIP):
            shoulder = skel.keypoints[KeypointIndex.LEFT_SHOULDER]
            hip = skel.keypoints[KeypointIndex.LEFT_HIP]
        else:
            shoulder = skel.keypoints[KeypointIndex.RIGHT_SHOULDER]
            hip = skel.keypoints[KeypointIndex.RIGHT_HIP]
        
        # 몸통 벡터 (엉덩이 → 어깨)
        torso_vec = vector_3d(hip, shoulder)
        
        # 수직 벡터 (카메라 좌표계에서 -Y가 위)
        vertical = np.array([0, -1, 0])
        
        # 몸통과 수직의 각도
        torso_norm = np.linalg.norm(torso_vec)
        if torso_norm < 1e-6:
            return False
        
        cos_angle = np.dot(torso_vec / torso_norm, vertical)
        torso_angle_from_vertical = np.degrees(np.arccos(np.clip(np.abs(cos_angle), 0, 1)))
        
        # 수직에서 45도 이상 벗어나면 (수평에 가까우면) 누워있음
        return torso_angle_from_vertical > (90 - self.lying_torso_horizontal_threshold)
    
    def _is_sitting(self, skel: Skeleton3D, angles: Dict) -> bool:
        """앉아있는 자세 감지"""
        # 무릎 각도 체크
        left_knee = angles.get("left_knee")
        right_knee = angles.get("right_knee")
        
        # 엉덩이 각도 체크
        left_hip = angles.get("left_hip")
        right_hip = angles.get("right_hip")
        
        # 무릎이 구부러져 있고 엉덩이도 구부러져 있으면 앉음
        knee_bent = False
        hip_bent = False
        
        if left_knee is not None and left_knee < self.sitting_knee_angle_max:
            knee_bent = True
        if right_knee is not None and right_knee < self.sitting_knee_angle_max:
            knee_bent = True
        
        if left_hip is not None and left_hip < self.sitting_hip_angle_max:
            hip_bent = True
        if right_hip is not None and right_hip < self.sitting_hip_angle_max:
            hip_bent = True
        
        return knee_bent and hip_bent
    
    def _is_standing(self, skel: Skeleton3D, angles: Dict) -> bool:
        """서있는 자세 감지"""
        # 무릎 각도 체크
        left_knee = angles.get("left_knee")
        right_knee = angles.get("right_knee")
        
        # 엉덩이 각도 체크
        left_hip = angles.get("left_hip")
        right_hip = angles.get("right_hip")
        
        # 무릎이 펴져 있고 엉덩이도 펴져 있으면 서있음
        knee_straight = False
        hip_straight = False
        
        if left_knee is not None and left_knee >= self.standing_knee_angle_min:
            knee_straight = True
        if right_knee is not None and right_knee >= self.standing_knee_angle_min:
            knee_straight = True
        
        if left_hip is not None and left_hip >= self.standing_hip_angle_min:
            hip_straight = True
        if right_hip is not None and right_hip >= self.standing_hip_angle_min:
            hip_straight = True
        
        # 둘 다 정보가 없으면 서있다고 가정
        if left_knee is None and right_knee is None:
            knee_straight = True
        if left_hip is None and right_hip is None:
            hip_straight = True
        
        return knee_straight and hip_straight


# ============================================================
# ROS2 Node
# ============================================================
class HumanStateDetectorNode(Node):
    def __init__(self):
        super().__init__('human_state_detector_node')
        
        # 파라미터 선언
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("world_frame", "base")
        
        self.declare_parameter("yolo_model", "yolov8s-pose.pt")
        self.declare_parameter("yolo_conf", 0.5)
        self.declare_parameter("yolo_device", "cuda")
        
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("min_keypoint_conf", 0.3)
        
        # 파라미터 로드
        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.world_frame = self.get_parameter("world_frame").value
        
        self.publish_markers = self.get_parameter("publish_markers").value
        self.min_keypoint_conf = self.get_parameter("min_keypoint_conf").value
        
        # QoS 설정
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        
        # CvBridge
        # self.bridge = CvBridge()
        
        # YOLO 모델 로드
        model_name = self.get_parameter("yolo_model").value
        device = self.get_parameter("yolo_device").value
        self.yolo_conf = self.get_parameter("yolo_conf").value
        
        self.get_logger().info(f"Loading YOLO model: {model_name} on {device}")
        self.model = YOLO(model_name)
        if device == 'cuda':
            self.model.to('cuda')
        
        # 카메라 정보
        self.camera_matrix = None
        self.dist_coeffs = None
        self.depth_image = None

        # 상태 플래그
        self.stop_flag = False
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 상태 감지기
        self.state_detector = HumanStateDetector()
        
        # 구독자
        self.sub_img = self.create_subscription(Image, self.image_topic, self.on_image, qos)
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_camera_info, 10)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, qos)

        # self.stop_flag 변경
        self.sub_flag = self.create_subscription(Bool, "/human/resume", self.flag_callback, 10)
        
        # 발행자
        self.pub_state = self.create_publisher(String, "/human/states", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/human/skeleton_markers", 10)
        self.pub_debug = self.create_publisher(Image, "/human/debug_image", 10)
        
        self.get_logger().info(f"Human State Detector Node initialized")
        self.get_logger().info(f"  Image: {self.image_topic}")
        self.get_logger().info(f"  Camera frame: {self.camera_frame}")
    
    def on_camera_info(self, msg: CameraInfo):
        """카메라 내부 파라미터 수신"""
        self.camera_matrix = np.array(msg.k).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d)
        self.destroy_subscription(self.sub_info)
        self.get_logger().info("Camera info received and camera matrix set.")
        
    
    def on_depth(self, msg: Image):
        """뎁스 이미지 수신"""
        try:
            self.depth_image = np.frombuffer(msg.data, dtype=np.uint16 if msg.encoding == "16UC1" else np.float32).reshape((msg.height, msg.width))
            # self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")
    
    def flag_callback(self, msg: Bool):
        if self.stop_flag != msg.data:
            rclpy.logging.get_logger("HumanStateDetectorNode").info(f"Setting stop_flag to {msg.data}")
        self.stop_flag = msg.data


    
    def pixel_to_3d(self, u: int, v: int, depth_m: float) -> Optional[Point3D]:
        """2D 픽셀 좌표 + 깊이 → 3D 좌표"""
        if self.camera_matrix is None:
            return None
        
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m
        
        return Point3D(x, y, z)
    
    def get_depth_at_pixel(self, u: int, v: int) -> Optional[float]:
        """픽셀 위치의 깊이값 (미터)"""
        if self.depth_image is None:
            return None
        
        h, w = self.depth_image.shape[:2]
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        
        y1, y2 = max(0, v - 2), min(h, v + 3)
        x1, x2 = max(0, u - 2), min(w, u + 3)
        patch = self.depth_image[y1:y2, x1:x2]
        
        valid = patch[(patch > 0) & np.isfinite(patch)]
        if valid.size == 0:
            return None
        
        depth_raw = float(np.median(valid))
        
        if self.depth_image.dtype == np.uint16:
            return depth_raw / 1000.0
        return depth_raw
    
    def extract_skeleton_3d(self, keypoints_2d: np.ndarray, img_shape: Tuple[int, int]) -> Skeleton3D:
        """2D 키포인트에서 3D 스켈레톤 추출"""
        h, w = img_shape[:2]
        skeleton = Skeleton3D(frame_id=self.camera_frame)
        
        for kp_idx in range(17):
            u, v, conf = keypoints_2d[kp_idx]
            
            if conf < self.min_keypoint_conf:
                continue
            
            u, v = int(u), int(v)
            if u < 0 or u >= w or v < 0 or v >= h:
                continue
            
            depth = self.get_depth_at_pixel(u, v)
            if depth is None or depth <= 0 or depth > 10.0:
                continue
            
            pt_3d = self.pixel_to_3d(u, v, depth)
            if pt_3d:
                pt_3d.confidence = float(conf)
                skeleton.keypoints[kp_idx] = pt_3d
        
        return skeleton
    
    def draw_state_on_image(self, image: np.ndarray, bbox: np.ndarray, 
                            state: HumanState, person_id: int,
                            angles: Dict[str, Optional[float]]) -> np.ndarray:
        """이미지에 상태 정보 그리기"""
        x1, y1, x2, y2 = map(int, bbox[:4])
        
        # 상태별 색상
        color = STATE_COLORS.get(state, (128, 128, 128))
        
        # 바운딩 박스
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # 상태 텍스트 배경
        state_text = f"P{person_id}: {state.name}"
        (text_w, text_h), _ = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), color, -1)
        
        # 상태 텍스트
        cv2.putText(image, state_text, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 각도 정보 (박스 아래)
        y_offset = y2 + 20
        for name, angle in angles.items():
            if angle is not None:
                angle_text = f"{name}: {angle:.1f}"
                cv2.putText(image, angle_text, (x1, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                y_offset += 15
                if y_offset > y2 + 100:  # 너무 많으면 자르기
                    break
        
        return image
    
    def create_skeleton_markers(self, skeleton: Skeleton3D, person_id: int, 
                                state: HumanState, stamp) -> MarkerArray:
        """스켈레톤 마커 생성"""
        markers = MarkerArray()
        
        # 상태별 색상
        color_bgr = STATE_COLORS.get(state, (128, 128, 128))
        color_rgb = (color_bgr[2] / 255.0, color_bgr[1] / 255.0, color_bgr[0] / 255.0)
        
        # 관절 점 마커
        joint_marker = Marker()
        joint_marker.header.frame_id = skeleton.frame_id
        joint_marker.header.stamp = stamp
        joint_marker.ns = f"skeleton_{person_id}_joints"
        joint_marker.id = 0
        joint_marker.type = Marker.SPHERE_LIST
        joint_marker.action = Marker.ADD
        joint_marker.scale.x = 0.05
        joint_marker.scale.y = 0.05
        joint_marker.scale.z = 0.05
        joint_marker.color.r = color_rgb[0]
        joint_marker.color.g = color_rgb[1]
        joint_marker.color.b = color_rgb[2]
        joint_marker.color.a = 1.0
        joint_marker.lifetime.sec = 0
        joint_marker.lifetime.nanosec = 200000000
        
        for idx, pt in skeleton.keypoints.items():
            joint_marker.points.append(pt.to_ros_point())
        
        markers.markers.append(joint_marker)
        
        # 뼈대 마커
        bone_marker = Marker()
        bone_marker.header.frame_id = skeleton.frame_id
        bone_marker.header.stamp = stamp
        bone_marker.ns = f"skeleton_{person_id}_bones"
        bone_marker.id = 1
        bone_marker.type = Marker.LINE_LIST
        bone_marker.action = Marker.ADD
        bone_marker.scale.x = 0.02
        bone_marker.color.r = color_rgb[0]
        bone_marker.color.g = color_rgb[1]
        bone_marker.color.b = color_rgb[2]
        bone_marker.color.a = 0.8
        bone_marker.lifetime.sec = 0
        bone_marker.lifetime.nanosec = 200000000
        
        for (start_idx, end_idx) in SKELETON_CONNECTIONS:
            if skeleton.has_point(start_idx) and skeleton.has_point(end_idx):
                bone_marker.points.append(skeleton.keypoints[start_idx].to_ros_point())
                bone_marker.points.append(skeleton.keypoints[end_idx].to_ros_point())
        
        markers.markers.append(bone_marker)
        
        # 상태 텍스트 마커
        center = skeleton.get_center()
        if center:
            text_marker = Marker()
            text_marker.header.frame_id = skeleton.frame_id
            text_marker.header.stamp = stamp
            text_marker.ns = f"skeleton_{person_id}_state"
            text_marker.id = 2
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position = center.to_ros_point()
            text_marker.pose.position.y -= 0.3  # 머리 위 (카메라 좌표계)
            text_marker.scale.z = 0.15
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            text_marker.text = state.name
            text_marker.lifetime.sec = 0
            text_marker.lifetime.nanosec = 200000000
            
            markers.markers.append(text_marker)
        
        return markers
    
    def on_image(self, msg: Image):
        """이미지 콜백"""
        if self.stop_flag:
            return
        
        
        try:
            bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
            # bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")
            return
        
        # YOLO 추론
        results = self.model(bgr, conf=self.yolo_conf, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            # 사람 없으면 원본 이미지 발행
            
            # debug_msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
            debug_msg = Image()
            debug_msg.data = bgr.tobytes()
            debug_msg.header = msg.header
            self.pub_debug.publish(debug_msg)
            return
        
        # 결과 이미지 (YOLO 기본 플롯)
        annotated = results[0].plot()
        
        keypoints_data = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else None
        
        all_markers = MarkerArray()
        all_states = []
        
        for person_idx, kp in enumerate(keypoints_data):
            # 3D 스켈레톤 추출
            skeleton = self.extract_skeleton_3d(kp, bgr.shape)
            
            if len(skeleton.keypoints) < 5:
                continue
            
            # 상태 판별
            state, angles = self.state_detector.detect_state(skeleton)
            all_states.append(f"P{person_idx}:{state.name}")
            
            # 디버그 이미지에 상태 그리기
            if boxes is not None and person_idx < len(boxes):
                annotated = self.draw_state_on_image(
                    annotated, boxes[person_idx], state, person_idx, angles
                )
            
            # 마커 생성
            if self.publish_markers:
                markers = self.create_skeleton_markers(skeleton, person_idx, state, msg.header.stamp)
                all_markers.markers.extend(markers.markers)
        
        # 상태 발행
        if all_states:
            state_msg = String()
            state_msg.data = ", ".join(all_states)
            self.pub_state.publish(state_msg)
        
        # 마커 발행
        if self.publish_markers and all_markers.markers:
            self.pub_markers.publish(all_markers)
        
        # 디버그 이미지 발행
        #debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        debug_msg = Image()
        debug_msg.data = annotated.tobytes()
        debug_msg.header = msg.header
        self.pub_debug.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HumanStateDetectorNode()
    
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
