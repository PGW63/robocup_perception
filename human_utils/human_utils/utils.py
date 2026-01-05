#!/usr/bin/env python3
"""
3D Human Skeleton Node
- YOLO Pose + Depth로 3D 스켈레톤 생성
- TF를 통해 로봇 좌표계로 변환
- RViz에서 MarkerArray로 시각화
- 세 점이 이루는 평면 상에서 각도 계산
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import tf2_ros
from tf2_ros import TransformBroadcaster


# ============================================================
# YOLO COCO Keypoint 인덱스 (17개)
# ============================================================
class KeypointIndex:
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    # 이름 매핑
    NAMES = {
        0: "nose", 1: "left_eye", 2: "right_eye", 3: "left_ear", 4: "right_ear",
        5: "left_shoulder", 6: "right_shoulder", 7: "left_elbow", 8: "right_elbow",
        9: "left_wrist", 10: "right_wrist", 11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee", 15: "left_ankle", 16: "right_ankle"
    }


# 스켈레톤 연결 (뼈대)
SKELETON_CONNECTIONS = [
    (KeypointIndex.NOSE, KeypointIndex.LEFT_EYE),
    (KeypointIndex.NOSE, KeypointIndex.RIGHT_EYE),
    (KeypointIndex.LEFT_EYE, KeypointIndex.LEFT_EAR),
    (KeypointIndex.RIGHT_EYE, KeypointIndex.RIGHT_EAR),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.RIGHT_SHOULDER),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_ELBOW),
    (KeypointIndex.LEFT_ELBOW, KeypointIndex.LEFT_WRIST),
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_ELBOW),
    (KeypointIndex.RIGHT_ELBOW, KeypointIndex.RIGHT_WRIST),
    (KeypointIndex.LEFT_SHOULDER, KeypointIndex.LEFT_HIP),
    (KeypointIndex.RIGHT_SHOULDER, KeypointIndex.RIGHT_HIP),
    (KeypointIndex.LEFT_HIP, KeypointIndex.RIGHT_HIP),
    (KeypointIndex.LEFT_HIP, KeypointIndex.LEFT_KNEE),
    (KeypointIndex.LEFT_KNEE, KeypointIndex.LEFT_ANKLE),
    (KeypointIndex.RIGHT_HIP, KeypointIndex.RIGHT_KNEE),
    (KeypointIndex.RIGHT_KNEE, KeypointIndex.RIGHT_ANKLE),
]


# ============================================================
# 3D Point & Skeleton 데이터 클래스
# ============================================================
@dataclass
class Point3D:
    x: float
    y: float
    z: float
    confidence: float = 1.0
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])
    
    def to_ros_point(self) -> Point:
        p = Point()
        p.x = float(self.x)
        p.y = float(self.y)
        p.z = float(self.z)
        return p
    
    @staticmethod
    def from_numpy(arr: np.ndarray, conf: float = 1.0) -> 'Point3D':
        return Point3D(float(arr[0]), float(arr[1]), float(arr[2]), conf)


@dataclass
class Skeleton3D:
    """3D 스켈레톤 (17개 keypoint)"""
    keypoints: Dict[int, Point3D] = field(default_factory=dict)
    person_id: int = 0
    timestamp: float = 0.0
    frame_id: str = "camera_link"
    
    def get_point(self, idx: int) -> Optional[Point3D]:
        return self.keypoints.get(idx)
    
    def has_point(self, idx: int, min_conf: float = 0.3) -> bool:
        pt = self.keypoints.get(idx)
        return pt is not None and pt.confidence >= min_conf
    
    def get_center(self) -> Optional[Point3D]:
        """스켈레톤 중심점 (hip 기준)"""
        if self.has_point(KeypointIndex.LEFT_HIP) and self.has_point(KeypointIndex.RIGHT_HIP):
            lh = self.keypoints[KeypointIndex.LEFT_HIP]
            rh = self.keypoints[KeypointIndex.RIGHT_HIP]
            return Point3D(
                (lh.x + rh.x) / 2,
                (lh.y + rh.y) / 2,
                (lh.z + rh.z) / 2,
                min(lh.confidence, rh.confidence)
            )
        return None


# ============================================================
# 3D 기하학 함수
# ============================================================
def angle_3d_on_plane(p1: Point3D, p2: Point3D, p3: Point3D) -> Tuple[float, np.ndarray]:
    """
    세 점이 이루는 평면 상에서의 각도 계산 (p2가 꼭지점)
    
    세 점 p1, p2, p3가 정의하는 평면 위에서,
    p2를 꼭지점으로 하는 ∠p1-p2-p3의 각도를 계산합니다.
    
    Args:
        p1: 첫 번째 점
        p2: 꼭지점 (각도를 측정할 점)
        p3: 세 번째 점
    
    Returns:
        Tuple[float, np.ndarray]: 
            - 각도 (도 단위, 0~180)
            - 평면의 법선 벡터 (단위 벡터)
    """
    # p2에서 p1, p3로 향하는 벡터
    v1 = p1.to_numpy() - p2.to_numpy()
    v2 = p3.to_numpy() - p2.to_numpy()
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0, np.array([0, 0, 1])
    
    # 단위 벡터
    v1_unit = v1 / norm1
    v2_unit = v2 / norm2
    
    # 평면의 법선 벡터 (v1 x v2)
    normal = np.cross(v1, v2)
    normal_norm = np.linalg.norm(normal)
    
    if normal_norm < 1e-6:
        # 세 점이 일직선상에 있음
        normal = np.array([0, 0, 1])
    else:
        normal = normal / normal_norm
    
    # 평면 상에서의 각도 (v1과 v2 사이)
    cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg, normal


def angle_3d(p1: Point3D, p2: Point3D, p3: Point3D) -> float:
    """
    세 점이 이루는 평면 상에서의 각도 계산 (p2가 꼭지점)
    
    Args:
        p1: 첫 번째 점
        p2: 꼭지점 (각도를 측정할 점)
        p3: 세 번째 점
    
    Returns:
        각도 (도 단위, 0~180)
    """
    angle, _ = angle_3d_on_plane(p1, p2, p3)
    return angle


def get_plane_from_3_points(p1: Point3D, p2: Point3D, p3: Point3D) -> Tuple[np.ndarray, float]:
    """
    세 점으로부터 평면의 방정식을 구함
    평면 방정식: ax + by + cz + d = 0
    
    Args:
        p1, p2, p3: 평면을 정의하는 세 점
    
    Returns:
        Tuple[np.ndarray, float]:
            - 법선 벡터 [a, b, c] (단위 벡터)
            - d 값
    """
    v1 = p2.to_numpy() - p1.to_numpy()
    v2 = p3.to_numpy() - p1.to_numpy()
    
    normal = np.cross(v1, v2)
    normal_norm = np.linalg.norm(normal)
    
    if normal_norm < 1e-6:
        # 세 점이 일직선상
        return np.array([0, 0, 1]), 0.0
    
    normal = normal / normal_norm
    d = -np.dot(normal, p1.to_numpy())
    
    return normal, d


def project_point_to_plane(point: Point3D, plane_normal: np.ndarray, plane_point: Point3D) -> Point3D:
    """
    점을 평면에 투영
    
    Args:
        point: 투영할 점
        plane_normal: 평면의 법선 벡터 (단위 벡터)
        plane_point: 평면 위의 한 점
    
    Returns:
        평면에 투영된 점
    """
    p = point.to_numpy()
    p0 = plane_point.to_numpy()
    n = plane_normal / np.linalg.norm(plane_normal)
    
    # 점에서 평면까지의 거리
    dist = np.dot(p - p0, n)
    
    # 투영된 점
    projected = p - dist * n
    
    return Point3D.from_numpy(projected, point.confidence)


def signed_angle_on_plane(p1: Point3D, p2: Point3D, p3: Point3D, 
                          reference_normal: Optional[np.ndarray] = None) -> float:
    """
    평면 상에서의 부호 있는 각도 계산 (p2가 꼭지점)
    
    reference_normal이 주어지면, 그 방향을 기준으로 시계/반시계 방향을 판단.
    
    Args:
        p1: 첫 번째 점
        p2: 꼭지점
        p3: 세 번째 점
        reference_normal: 참조 법선 벡터 (없으면 자동 계산)
    
    Returns:
        부호 있는 각도 (도 단위, -180 ~ 180)
        양수: p1에서 p3로 반시계 방향 (reference_normal 기준)
        음수: 시계 방향
    """
    v1 = p1.to_numpy() - p2.to_numpy()
    v2 = p3.to_numpy() - p2.to_numpy()
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    v1_unit = v1 / norm1
    v2_unit = v2 / norm2
    
    # 평면 법선
    if reference_normal is None:
        reference_normal = np.cross(v1, v2)
        ref_norm = np.linalg.norm(reference_normal)
        if ref_norm < 1e-6:
            return 0.0
        reference_normal = reference_normal / ref_norm
    
    # 무부호 각도
    cos_angle = np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    
    # 부호 결정 (외적의 방향과 참조 법선 비교)
    cross = np.cross(v1_unit, v2_unit)
    if np.dot(cross, reference_normal) < 0:
        angle = -angle
    
    return angle


def distance_3d(p1: Point3D, p2: Point3D) -> float:
    """두 점 사이의 3D 거리"""
    return np.linalg.norm(p1.to_numpy() - p2.to_numpy())


def vector_3d(p1: Point3D, p2: Point3D) -> np.ndarray:
    """p1에서 p2로의 벡터"""
    return p2.to_numpy() - p1.to_numpy()


def midpoint_3d(p1: Point3D, p2: Point3D) -> Point3D:
    """두 점의 중점"""
    return Point3D(
        (p1.x + p2.x) / 2,
        (p1.y + p2.y) / 2,
        (p1.z + p2.z) / 2,
        min(p1.confidence, p2.confidence)
    )


# ============================================================
# 관절 각도 계산 유틸리티
# ============================================================
class JointAngleCalculator:
    """스켈레톤의 주요 관절 각도 계산"""
    
    @staticmethod
    def left_elbow_angle(skel: Skeleton3D) -> Optional[float]:
        """왼쪽 팔꿈치 각도 (어깨-팔꿈치-손목)"""
        if not all(skel.has_point(i) for i in [
            KeypointIndex.LEFT_SHOULDER,
            KeypointIndex.LEFT_ELBOW,
            KeypointIndex.LEFT_WRIST
        ]):
            return None
        return angle_3d(
            skel.keypoints[KeypointIndex.LEFT_SHOULDER],
            skel.keypoints[KeypointIndex.LEFT_ELBOW],
            skel.keypoints[KeypointIndex.LEFT_WRIST]
        )
    
    @staticmethod
    def right_elbow_angle(skel: Skeleton3D) -> Optional[float]:
        """오른쪽 팔꿈치 각도 (어깨-팔꿈치-손목)"""
        if not all(skel.has_point(i) for i in [
            KeypointIndex.RIGHT_SHOULDER,
            KeypointIndex.RIGHT_ELBOW,
            KeypointIndex.RIGHT_WRIST
        ]):
            return None
        return angle_3d(
            skel.keypoints[KeypointIndex.RIGHT_SHOULDER],
            skel.keypoints[KeypointIndex.RIGHT_ELBOW],
            skel.keypoints[KeypointIndex.RIGHT_WRIST]
        )
    
    @staticmethod
    def left_shoulder_angle(skel: Skeleton3D) -> Optional[float]:
        """왼쪽 어깨 각도 (엉덩이-어깨-팔꿈치)"""
        if not all(skel.has_point(i) for i in [
            KeypointIndex.LEFT_HIP,
            KeypointIndex.LEFT_SHOULDER,
            KeypointIndex.LEFT_ELBOW
        ]):
            return None
        return angle_3d(
            skel.keypoints[KeypointIndex.LEFT_HIP],
            skel.keypoints[KeypointIndex.LEFT_SHOULDER],
            skel.keypoints[KeypointIndex.LEFT_ELBOW]
        )
    
    @staticmethod
    def right_shoulder_angle(skel: Skeleton3D) -> Optional[float]:
        """오른쪽 어깨 각도 (엉덩이-어깨-팔꿈치)"""
        if not all(skel.has_point(i) for i in [
            KeypointIndex.RIGHT_HIP,
            KeypointIndex.RIGHT_SHOULDER,
            KeypointIndex.RIGHT_ELBOW
        ]):
            return None
        return angle_3d(
            skel.keypoints[KeypointIndex.RIGHT_HIP],
            skel.keypoints[KeypointIndex.RIGHT_SHOULDER],
            skel.keypoints[KeypointIndex.RIGHT_ELBOW]
        )
    
    @staticmethod
    def left_knee_angle(skel: Skeleton3D) -> Optional[float]:
        """왼쪽 무릎 각도 (엉덩이-무릎-발목)"""
        if not all(skel.has_point(i) for i in [
            KeypointIndex.LEFT_HIP,
            KeypointIndex.LEFT_KNEE,
            KeypointIndex.LEFT_ANKLE
        ]):
            return None
        return angle_3d(
            skel.keypoints[KeypointIndex.LEFT_HIP],
            skel.keypoints[KeypointIndex.LEFT_KNEE],
            skel.keypoints[KeypointIndex.LEFT_ANKLE]
        )
    
    @staticmethod
    def right_knee_angle(skel: Skeleton3D) -> Optional[float]:
        """오른쪽 무릎 각도 (엉덩이-무릎-발목)"""
        if not all(skel.has_point(i) for i in [
            KeypointIndex.RIGHT_HIP,
            KeypointIndex.RIGHT_KNEE,
            KeypointIndex.RIGHT_ANKLE
        ]):
            return None
        return angle_3d(
            skel.keypoints[KeypointIndex.RIGHT_HIP],
            skel.keypoints[KeypointIndex.RIGHT_KNEE],
            skel.keypoints[KeypointIndex.RIGHT_ANKLE]
        )
    
    @staticmethod
    def left_hip_angle(skel: Skeleton3D) -> Optional[float]:
        """왼쪽 엉덩이 각도 (어깨-엉덩이-무릎)"""
        if not all(skel.has_point(i) for i in [
            KeypointIndex.LEFT_SHOULDER,
            KeypointIndex.LEFT_HIP,
            KeypointIndex.LEFT_KNEE
        ]):
            return None
        return angle_3d(
            skel.keypoints[KeypointIndex.LEFT_SHOULDER],
            skel.keypoints[KeypointIndex.LEFT_HIP],
            skel.keypoints[KeypointIndex.LEFT_KNEE]
        )
    
    @staticmethod
    def right_hip_angle(skel: Skeleton3D) -> Optional[float]:
        """오른쪽 엉덩이 각도 (어깨-엉덩이-무릎)"""
        if not all(skel.has_point(i) for i in [
            KeypointIndex.RIGHT_SHOULDER,
            KeypointIndex.RIGHT_HIP,
            KeypointIndex.RIGHT_KNEE
        ]):
            return None
        return angle_3d(
            skel.keypoints[KeypointIndex.RIGHT_SHOULDER],
            skel.keypoints[KeypointIndex.RIGHT_HIP],
            skel.keypoints[KeypointIndex.RIGHT_KNEE]
        )
    
    @staticmethod
    def get_all_angles(skel: Skeleton3D) -> Dict[str, Optional[float]]:
        """모든 주요 관절 각도 반환"""
        calc = JointAngleCalculator
        return {
            "left_elbow": calc.left_elbow_angle(skel),
            "right_elbow": calc.right_elbow_angle(skel),
            "left_shoulder": calc.left_shoulder_angle(skel),
            "right_shoulder": calc.right_shoulder_angle(skel),
            "left_knee": calc.left_knee_angle(skel),
            "right_knee": calc.right_knee_angle(skel),
            "left_hip": calc.left_hip_angle(skel),
            "right_hip": calc.right_hip_angle(skel),
        }


# # ============================================================
# # ROS2 Node
# # ============================================================
# class HumanSkeleton3DNode(Node):
#     def __init__(self):
#         super().__init__('human_skeleton_3d_node')
        
#         # 파라미터 선언
#         self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
#         self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
#         self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
#         self.declare_parameter("camera_frame", "camera_color_optical_frame")
#         self.declare_parameter("world_frame", "base_link")
        
#         self.declare_parameter("yolo_model", "yolov8n-pose.pt")
#         self.declare_parameter("yolo_conf", 0.5)
#         self.declare_parameter("yolo_device", "cuda")
        
#         self.declare_parameter("publish_markers", True)
#         self.declare_parameter("min_keypoint_conf", 0.3)
        
#         # 파라미터 로드
#         self.image_topic = self.get_parameter("image_topic").value
#         self.camera_info_topic = self.get_parameter("camera_info_topic").value
#         self.depth_topic = self.get_parameter("depth_topic").value
#         self.camera_frame = self.get_parameter("camera_frame").value
#         self.world_frame = self.get_parameter("world_frame").value
        
#         self.publish_markers = self.get_parameter("publish_markers").value
#         self.min_keypoint_conf = self.get_parameter("min_keypoint_conf").value
        
#         # QoS 설정
#         qos = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=5,
#         )
        
#         # CvBridge
#         self.bridge = CvBridge()
        
#         # YOLO 모델 로드
#         model_name = self.get_parameter("yolo_model").value
#         device = self.get_parameter("yolo_device").value
#         self.yolo_conf = self.get_parameter("yolo_conf").value
        
#         self.get_logger().info(f"Loading YOLO model: {model_name} on {device}")
#         self.model = YOLO(model_name)
#         if device == 'cuda':
#             self.model.to('cuda')
        
#         # 카메라 정보
#         self.camera_matrix = None
#         self.dist_coeffs = None
#         self.depth_image = None
        
#         # TF
#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
#         self.tf_broadcaster = TransformBroadcaster(self)
        
#         # 구독자
#         self.sub_img = self.create_subscription(Image, self.image_topic, self.on_image, qos)
#         self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.on_camera_info, 10)
#         self.sub_depth = self.create_subscription(Image, self.depth_topic, self.on_depth, qos)
        
#         # 발행자
#         self.pub_markers = self.create_publisher(MarkerArray, "/human/skeleton_markers", 10)
#         self.pub_debug = self.create_publisher(Image, "/human/debug_image", 10)
        
#         self.get_logger().info(f"Human Skeleton 3D Node initialized")
#         self.get_logger().info(f"  Image: {self.image_topic}")
#         self.get_logger().info(f"  Camera frame: {self.camera_frame}")
#         self.get_logger().info(f"  World frame: {self.world_frame}")
    
#     def on_camera_info(self, msg: CameraInfo):
#         """카메라 내부 파라미터 수신"""
#         self.camera_matrix = np.array(msg.k).reshape((3, 3))
#         self.dist_coeffs = np.array(msg.d)
    
#     def on_depth(self, msg: Image):
#         """뎁스 이미지 수신"""
#         try:
#             self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
#         except Exception as e:
#             self.get_logger().warn(f"Depth conversion failed: {e}")
    
#     def pixel_to_3d(self, u: int, v: int, depth_m: float) -> Optional[Point3D]:
#         """2D 픽셀 좌표 + 깊이 → 3D 좌표 (카메라 좌표계)"""
#         if self.camera_matrix is None:
#             return None
        
#         fx = self.camera_matrix[0, 0]
#         fy = self.camera_matrix[1, 1]
#         cx = self.camera_matrix[0, 2]
#         cy = self.camera_matrix[1, 2]
        
#         # 카메라 좌표계로 변환
#         x = (u - cx) * depth_m / fx
#         y = (v - cy) * depth_m / fy
#         z = depth_m
        
#         return Point3D(x, y, z)
    
#     def get_depth_at_pixel(self, u: int, v: int) -> Optional[float]:
#         """픽셀 위치의 깊이값 (미터)"""
#         if self.depth_image is None:
#             return None
        
#         h, w = self.depth_image.shape[:2]
#         u = int(np.clip(u, 0, w - 1))
#         v = int(np.clip(v, 0, h - 1))
        
#         # 주변 영역 샘플링 (노이즈 감소)
#         y1, y2 = max(0, v - 2), min(h, v + 3)
#         x1, x2 = max(0, u - 2), min(w, u + 3)
#         patch = self.depth_image[y1:y2, x1:x2]
        
#         valid = patch[(patch > 0) & np.isfinite(patch)]
#         if valid.size == 0:
#             return None
        
#         depth_raw = float(np.median(valid))
        
#         # uint16이면 mm, float이면 m
#         if self.depth_image.dtype == np.uint16:
#             return depth_raw / 1000.0
#         return depth_raw
    
#     def extract_skeleton_3d(self, results, img_shape: Tuple[int, int]) -> List[Skeleton3D]:
#         """YOLO 결과에서 3D 스켈레톤 추출"""
#         skeletons = []
        
#         if len(results) == 0 or results[0].keypoints is None:
#             return skeletons
        
#         h, w = img_shape[:2]
#         keypoints_data = results[0].keypoints.data.cpu().numpy()  # (N, 17, 3)
        
#         for person_idx, kp in enumerate(keypoints_data):
#             skeleton = Skeleton3D(person_id=person_idx, frame_id=self.camera_frame)
            
#             for kp_idx in range(17):
#                 u, v, conf = kp[kp_idx]
                
#                 if conf < self.min_keypoint_conf:
#                     continue
                
#                 u, v = int(u), int(v)
#                 if u < 0 or u >= w or v < 0 or v >= h:
#                     continue
                
#                 depth = self.get_depth_at_pixel(u, v)
#                 if depth is None or depth <= 0 or depth > 10.0:
#                     continue
                
#                 pt_3d = self.pixel_to_3d(u, v, depth)
#                 if pt_3d:
#                     pt_3d.confidence = float(conf)
#                     skeleton.keypoints[kp_idx] = pt_3d
            
#             if len(skeleton.keypoints) >= 5:  # 최소 5개 키포인트
#                 skeletons.append(skeleton)
        
#         return skeletons
    
#     def transform_skeleton_to_world(self, skeleton: Skeleton3D) -> Optional[Skeleton3D]:
#         """스켈레톤을 월드 좌표계로 변환"""
#         try:
#             transform = self.tf_buffer.lookup_transform(
#                 self.world_frame,
#                 self.camera_frame,
#                 rclpy.time.Time(),
#                 timeout=rclpy.duration.Duration(seconds=0.1)
#             )
#         except Exception as e:
#             # TF 없으면 카메라 좌표계 그대로 사용
#             return skeleton
        
#         world_skeleton = Skeleton3D(
#             person_id=skeleton.person_id,
#             frame_id=self.world_frame
#         )
        
#         # 변환 행렬 생성
#         t = transform.transform.translation
#         r = transform.transform.rotation
        
#         # 쿼터니언 → 회전 행렬
#         from scipy.spatial.transform import Rotation
#         rot = Rotation.from_quat([r.x, r.y, r.z, r.w])
#         rot_matrix = rot.as_matrix()
        
#         for idx, pt in skeleton.keypoints.items():
#             pt_cam = pt.to_numpy()
#             pt_world = rot_matrix @ pt_cam + np.array([t.x, t.y, t.z])
#             world_skeleton.keypoints[idx] = Point3D.from_numpy(pt_world, pt.confidence)
        
#         return world_skeleton
    
#     def create_skeleton_markers(self, skeleton: Skeleton3D, stamp) -> MarkerArray:
#         """스켈레톤을 RViz MarkerArray로 변환"""
#         markers = MarkerArray()
        
#         # 1. 관절 점 마커 (Sphere)
#         joint_marker = Marker()
#         joint_marker.header.frame_id = skeleton.frame_id
#         joint_marker.header.stamp = stamp
#         joint_marker.ns = f"skeleton_{skeleton.person_id}_joints"
#         joint_marker.id = 0
#         joint_marker.type = Marker.SPHERE_LIST
#         joint_marker.action = Marker.ADD
#         joint_marker.scale.x = 0.05  # 5cm 구
#         joint_marker.scale.y = 0.05
#         joint_marker.scale.z = 0.05
#         joint_marker.color.r = 0.0
#         joint_marker.color.g = 1.0
#         joint_marker.color.b = 0.0
#         joint_marker.color.a = 1.0
#         joint_marker.lifetime.sec = 0
#         joint_marker.lifetime.nanosec = 200000000  # 200ms
        
#         for idx, pt in skeleton.keypoints.items():
#             joint_marker.points.append(pt.to_ros_point())
#             # 신뢰도에 따른 색상
#             color = ColorRGBA()
#             color.r = 1.0 - pt.confidence
#             color.g = pt.confidence
#             color.b = 0.0
#             color.a = 1.0
#             joint_marker.colors.append(color)
        
#         markers.markers.append(joint_marker)
        
#         # 2. 뼈대 연결 마커 (Line List)
#         bone_marker = Marker()
#         bone_marker.header.frame_id = skeleton.frame_id
#         bone_marker.header.stamp = stamp
#         bone_marker.ns = f"skeleton_{skeleton.person_id}_bones"
#         bone_marker.id = 1
#         bone_marker.type = Marker.LINE_LIST
#         bone_marker.action = Marker.ADD
#         bone_marker.scale.x = 0.02  # 2cm 선
#         bone_marker.color.r = 0.0
#         bone_marker.color.g = 0.8
#         bone_marker.color.b = 1.0
#         bone_marker.color.a = 0.8
#         bone_marker.lifetime.sec = 0
#         bone_marker.lifetime.nanosec = 200000000
        
#         for (start_idx, end_idx) in SKELETON_CONNECTIONS:
#             if skeleton.has_point(start_idx) and skeleton.has_point(end_idx):
#                 bone_marker.points.append(skeleton.keypoints[start_idx].to_ros_point())
#                 bone_marker.points.append(skeleton.keypoints[end_idx].to_ros_point())
        
#         markers.markers.append(bone_marker)
        
#         # 3. 관절 각도 텍스트 마커
#         angles = JointAngleCalculator.get_all_angles(skeleton)
#         center = skeleton.get_center()
        
#         if center:
#             # 각도 정보 텍스트
#             angle_text = []
#             for name, angle in angles.items():
#                 if angle is not None:
#                     angle_text.append(f"{name}: {angle:.1f}")
            
#             if angle_text:
#                 text_marker = Marker()
#                 text_marker.header.frame_id = skeleton.frame_id
#                 text_marker.header.stamp = stamp
#                 text_marker.ns = f"skeleton_{skeleton.person_id}_angles"
#                 text_marker.id = 2
#                 text_marker.type = Marker.TEXT_VIEW_FACING
#                 text_marker.action = Marker.ADD
#                 text_marker.pose.position = center.to_ros_point()
#                 text_marker.pose.position.z += 0.5  # 머리 위
#                 text_marker.scale.z = 0.08  # 텍스트 크기
#                 text_marker.color.r = 1.0
#                 text_marker.color.g = 1.0
#                 text_marker.color.b = 0.0
#                 text_marker.color.a = 1.0
#                 text_marker.text = "\n".join(angle_text[:4])  # 상위 4개만
#                 text_marker.lifetime.sec = 0
#                 text_marker.lifetime.nanosec = 200000000
                
#                 markers.markers.append(text_marker)
        
#         return markers
    
#     def on_image(self, msg: Image):
#         """이미지 콜백"""
#         try:
#             bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
#         except Exception as e:
#             self.get_logger().warn(f"Image conversion failed: {e}")
#             return
        
#         # YOLO 추론
#         results = self.model(bgr, conf=self.yolo_conf, verbose=False)
        
#         # 3D 스켈레톤 추출
#         skeletons = self.extract_skeleton_3d(results, bgr.shape)
        
#         all_markers = MarkerArray()
        
#         for skeleton in skeletons:
#             # 월드 좌표계로 변환 (TF 있으면)
#             world_skeleton = self.transform_skeleton_to_world(skeleton)
            
#             # 마커 생성 & 발행
#             if self.publish_markers:
#                 markers = self.create_skeleton_markers(world_skeleton or skeleton, msg.header.stamp)
#                 all_markers.markers.extend(markers.markers)
            
#             # 관절 각도 계산 (로그 출력용)
#             angles = JointAngleCalculator.get_all_angles(world_skeleton or skeleton)
#             # 필요시 여기서 상태 판별 로직 추가 가능
        
#         if self.publish_markers and all_markers.markers:
#             self.pub_markers.publish(all_markers)
        
#         # 디버그 이미지 발행
#         annotated = results[0].plot()
        
#         # 각도 정보 표시
#         y_offset = 30
#         for skeleton in skeletons:
#             angles = JointAngleCalculator.get_all_angles(skeleton)
#             for name, angle in angles.items():
#                 if angle is not None:
#                     cv2.putText(annotated, f"{name}: {angle:.1f}", (20, y_offset),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
#                     y_offset += 20
        
#         debug_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
#         debug_msg.header = msg.header
#         self.pub_debug.publish(debug_msg)


# def main(args=None):
#     rclpy.init(args=args)
#     node = HumanSkeleton3DNode()
    
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()


# if __name__ == "__main__":
#     main()
