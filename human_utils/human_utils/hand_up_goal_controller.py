#!/usr/bin/env python3
"""
Hand Up Goal Controller Node (자동 제어 버전)
- HANDS_UP 상태의 사람 감지
- 가장 가까운 스켈레톤 점 기준 1.2m 앞 목적지 계산
- map 프레임으로 변환하여 nav2 목적지 발행
- rviz 시각화
- 거리 모니터링하여 임계값 이하면 navigation 자동 취소
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from nav2_msgs.action import NavigateToPose
import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs
from typing import Dict, List, Optional, Tuple
from rclpy.callback_groups import ReentrantCallbackGroup


class HandUpGoalController(Node):
    def __init__(self):
        super().__init__('hand_up_goal_controller')
        
        # 파라미터 선언
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("goal_distance", 1.2)  # 목적지까지 거리 (미터)
        self.declare_parameter("min_skeleton_points", 5)  # 최소 스켈레톤 점 개수
        self.declare_parameter("stop_distance", 0.2)  # 정지 거리 임계값 (미터)
        self.declare_parameter("distance_check_rate", 5.0)  # 거리 체크 주기 (Hz)
        self.declare_parameter("use_nav2", True)  # nav2 action 사용 여부
        
        # 강건성 파라미터
        self.declare_parameter("min_detection_frames", 5)  # 최소 연속 감지 프레임 수
        
        # 파라미터 로드
        self.map_frame = self.get_parameter("map_frame").value
        self.goal_distance = self.get_parameter("goal_distance").value
        self.min_skeleton_points = self.get_parameter("min_skeleton_points").value
        self.stop_distance = self.get_parameter("stop_distance").value
        self.distance_check_rate = self.get_parameter("distance_check_rate").value
        self.use_nav2 = self.get_parameter("use_nav2").value
        
        self.min_detection_frames = self.get_parameter("min_detection_frames").value
        
        # Callback group (action과 subscription이 동시에 동작하도록)
        self.callback_group = ReentrantCallbackGroup()
        
        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # 상태 저장
        self.current_states = {}  # {person_id: state_name}
        self.skeleton_data = {}   # {person_id: {'frame_id': str, 'points': [(x, y, z), ...]}}
        
        # Navigation 상태
        self.current_target_person_id = None  # 현재 목표로 하는 사람 ID
        self.navigation_active = False  # navigation이 활성화되어 있는지
        self.goal_handle = None  # 현재 navigation goal handle
        
        # 강건성 검증
        self.detection_counter = {}  # {person_id: count}
        self.goal_sent_for_person = set()  # 이미 goal을 보낸 사람 ID들
        
        # QoS 설정
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        
        # 구독자
        self.sub_states = self.create_subscription(
            String, "/human/states", self.on_states, 10,
            callback_group=self.callback_group
        )
        self.sub_markers = self.create_subscription(
            MarkerArray, "/human/skeleton_markers", self.on_skeleton_markers, qos,
            callback_group=self.callback_group
        )
        # 리셋 토픽
        self.sub_reset = self.create_subscription(
            String, "/hand_up_goal/reset", self.on_reset, 10,
            callback_group=self.callback_group
        )
        
        # 발행자
        self.pub_goal = self.create_publisher(PoseStamped, "/hand_up_goal", 10)
        self.pub_goal_marker = self.create_publisher(Marker, "/hand_up_goal_marker", 10)
        
        # Nav2 Action Client
        if self.use_nav2:
            self.nav2_client = ActionClient(
                self, 
                NavigateToPose, 
                'navigate_to_pose',
                callback_group=self.callback_group
            )
            self.get_logger().info("Waiting for nav2 action server...")
            # 비동기로 대기 (블록하지 않음)
        
        # 거리 모니터링 타이머
        if self.distance_check_rate > 0:
            self.distance_timer = self.create_timer(
                1.0 / self.distance_check_rate,
                self.check_distance_and_stop,
                callback_group=self.callback_group
            )
        
        self.get_logger().info("Hand Up Goal Controller Node initialized (Auto-Control Mode)")
        self.get_logger().info(f"  Map frame: {self.map_frame}")
        self.get_logger().info(f"  Goal distance: {self.goal_distance}m")
        self.get_logger().info(f"  Stop distance: {self.stop_distance}m")
        self.get_logger().info(f"  Use nav2: {self.use_nav2}")
        self.get_logger().info(f"  Min detection frames: {self.min_detection_frames} (연속 감지 필요)")
        self.get_logger().info(f"  Mode: Auto-cancel navigation when reaching threshold")
        self.get_logger().info(f"  Reset topic: /hand_up_goal/reset (publish any string to reset)")
        self.get_logger().info(f"  Reset topic: /hand_up_goal/reset (publish any string to reset)")
    
    def on_reset(self, msg: String):
        """리셋 토픽 수신"""
        # 현재 navigation 취소
        if self.navigation_active:
            self.cancel_navigation()
        
        self.detection_counter.clear()
        self.goal_sent_for_person.clear()
        self.current_target_person_id = None
        self.get_logger().info("🔄 Goal system RESET! Ready to detect new hand-up gesture.")
    
    def on_states(self, msg: String):
        """사람 상태 수신 (예: "P0:HAND_UP_LEFT, P1:STANDING")"""
        self.current_states.clear()
        
        if not msg.data:
            return
        
        # 상태 파싱
        for person_state in msg.data.split(", "):
            try:
                person_id_str, state_name = person_state.split(":")
                person_id = int(person_id_str[1:])  # "P0" -> 0
                self.current_states[person_id] = state_name
            except (ValueError, IndexError) as e:
                self.get_logger().warn(f"Failed to parse state: {person_state}, error: {e}")
    
    def on_skeleton_markers(self, msg: MarkerArray):
        """스켈레톤 마커 수신"""
        self.skeleton_data.clear()
        
        # 마커에서 스켈레톤 점들 추출
        for marker in msg.markers:
            # "skeleton_{person_id}_joints" 형태의 namespace
            if "_joints" not in marker.ns:
                continue
            
            try:
                person_id = int(marker.ns.split("_")[1])
            except (ValueError, IndexError):
                continue
            
            # SPHERE_LIST 타입 마커에서 점들 추출
            if marker.type == Marker.SPHERE_LIST:
                points = []
                for pt in marker.points:
                    points.append((pt.x, pt.y, pt.z))
                
                if len(points) >= self.min_skeleton_points:
                    # 마커의 frame_id 저장
                    self.skeleton_data[person_id] = {
                        'frame_id': marker.header.frame_id,
                        'points': points
                    }
        
        # HAND_UP 상태인 사람 찾아서 목적지 계산
        self.process_hand_up_goals()
    
    def process_hand_up_goals(self):
        """HAND_UP 상태인 사람들의 목적지 계산 및 발행"""
        hand_up_states = ["HAND_UP_LEFT", "HAND_UP_RIGHT", "HAND_UP_BOTH"]
        
        # 현재 HAND_UP 상태인 사람들
        current_hand_up_persons = []
        for person_id, state in self.current_states.items():
            if state in hand_up_states and person_id in self.skeleton_data:
                # 이미 goal을 보낸 사람은 제외
                if person_id not in self.goal_sent_for_person:
                    current_hand_up_persons.append(person_id)
        
        # 감지 카운터 업데이트
        persons_to_remove = []
        for person_id in self.detection_counter.keys():
            if person_id not in current_hand_up_persons:
                persons_to_remove.append(person_id)
        for person_id in persons_to_remove:
            del self.detection_counter[person_id]
        
        if not current_hand_up_persons:
            return
        
        # 가장 가까운 사람 선택
        closest_person_id = None
        min_distance = float('inf')
        
        for person_id in current_hand_up_persons:
            skeleton_info = self.skeleton_data[person_id]
            points = skeleton_info['points']
            distances = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in points]
            closest_dist = min(distances)
            
            if closest_dist < min_distance:
                min_distance = closest_dist
                closest_person_id = person_id
        
        if closest_person_id is None:
            return
        
        # 연속 감지 카운터 증가
        if closest_person_id not in self.detection_counter:
            self.detection_counter[closest_person_id] = 0
        self.detection_counter[closest_person_id] += 1
        
        # 최소 프레임 수만큼 연속 감지되지 않았으면 대기
        if self.detection_counter[closest_person_id] < self.min_detection_frames:
            self.get_logger().debug(
                f"⏳ Detecting person {closest_person_id}: "
                f"{self.detection_counter[closest_person_id]}/{self.min_detection_frames} frames"
            )
            return
        
        self.get_logger().info(
            f"✅ Person {closest_person_id} consistently detected! Sending goal."
        )
        
        # 가장 가까운 점 찾기
        skeleton_info = self.skeleton_data[closest_person_id]
        points = skeleton_info['points']
        skeleton_frame = skeleton_info['frame_id']
        distances = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in points]
        closest_idx = np.argmin(distances)
        closest_point = points[closest_idx]
        
        self.get_logger().info(
            f"Person {closest_person_id} ({self.current_states[closest_person_id]}) "
            f"closest point: ({closest_point[0]:.2f}, {closest_point[1]:.2f}, {closest_point[2]:.2f}) "
            f"in frame: {skeleton_frame}"
        )
        
        # 1.2m 앞 목적지 계산 (스켈레톤 프레임에서)
        goal_skeleton_frame = self.calculate_goal_in_front(closest_point, self.goal_distance)
        
        if goal_skeleton_frame is None:
            return
        
        # map 프레임으로 변환
        goal_map_frame = self.transform_to_map(goal_skeleton_frame, skeleton_frame)
        
        if goal_map_frame is None:
            return
        
        # 목적지 발행
        self.publish_goal(goal_map_frame, closest_person_id)
        
        # 이 사람에게 이미 goal을 보냈다고 기록
        self.goal_sent_for_person.add(closest_person_id)
    
    def calculate_goal_in_front(self, closest_point: Tuple[float, float, float], 
                                 distance: float) -> Optional[Tuple[float, float, float]]:
        """
        가장 가까운 점에서 원점(로봇/카메라)을 향해 distance 미터 앞 좌표 계산
        
        Args:
            closest_point: 가장 가까운 스켈레톤 점 (x, y, z)
            distance: 목적지까지 거리 (미터)
        
        Returns:
            목적지 좌표 (x, y, z) 또는 None
        """
        x, y, z = closest_point
        
        # 원점에서 사람으로의 방향 벡터
        direction = np.array([x, y, z])
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 0.1:  # 너무 가까우면 무시
            return None
        
        # 정규화
        direction_unit = direction / direction_norm
        
        # 사람 위치에서 원점(로봇) 방향으로 distance만큼 이동
        # (사람 - distance * 방향)
        goal = direction - distance * direction_unit
        
        return (goal[0], goal[1], goal[2])
    
    def transform_to_map(self, point: Tuple[float, float, float], 
                        source_frame: str) -> Optional[Tuple[float, float, float, float]]:
        """
        source_frame의 점을 map 좌표계로 변환
        
        Returns:
            (x, y, z, yaw) in map frame, or None if transform failed
        """
        try:
            # TF 조회
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed ({source_frame} -> {self.map_frame}): {e}")
            return None
        
        # PoseStamped 생성
        pose_source = PoseStamped()
        pose_source.header.frame_id = source_frame
        pose_source.header.stamp = self.get_clock().now().to_msg()
        pose_source.pose.position.x = point[0]
        pose_source.pose.position.y = point[1]
        pose_source.pose.position.z = point[2]
        pose_source.pose.orientation.w = 1.0
        
        # 변환
        try:
            pose_map = tf2_geometry_msgs.do_transform_pose_stamped(pose_source, transform)
        except Exception as e:
            self.get_logger().warn(f"Transform failed: {e}")
            return None
        
        # yaw 계산 (목적지에서 사람을 바라보는 방향)
        # 사람 위치도 map 좌표로 변환 필요
        # 하지만 여기서는 단순히 목적지만 변환
        # yaw는 나중에 사람 위치를 보고 계산
        
        x = pose_map.pose.position.x
        y = pose_map.pose.position.y
        z = pose_map.pose.position.z
        
        # yaw는 0으로 설정 (나중에 계산 가능)
        yaw = 0.0
        
        return (x, y, z, yaw)
    
    def publish_goal(self, goal_map: Tuple[float, float, float, float], person_id: int):
        """nav2 목적지 및 시각화 마커 발행"""
        x, y, z, yaw = goal_map
        
        # PoseStamped 생성
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = self.map_frame
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.position.x = x
        goal_msg.pose.position.y = y
        goal_msg.pose.position.z = 0.0  # 지면 기준
        
        # Quaternion (yaw만 설정)
        goal_msg.pose.orientation.z = np.sin(yaw / 2.0)
        goal_msg.pose.orientation.w = np.cos(yaw / 2.0)
        
        # PoseStamped 발행 (rviz 등에서 사용)
        self.pub_goal.publish(goal_msg)
        
        self.get_logger().info(
            f"Published goal for person {person_id}: "
            f"({x:.2f}, {y:.2f}, {z:.2f}) in {self.map_frame}"
        )
        
        # Nav2 Action으로 goal 전송
        if self.use_nav2:
            self.send_nav2_goal(goal_msg, person_id)
        
        # 시각화 마커 발행
        self.publish_goal_marker(goal_map, person_id)
    
    def publish_goal_marker(self, goal_map: Tuple[float, float, float, float], person_id: int):
        """rviz 시각화 마커 발행"""
        x, y, z, yaw = goal_map
        
        # 화살표 마커 (목적지 표시)
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "hand_up_goal"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # 위치
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0  # 지면
        
        # 방향 (yaw)
        marker.pose.orientation.z = np.sin(yaw / 2.0)
        marker.pose.orientation.w = np.cos(yaw / 2.0)
        
        # 크기
        marker.scale.x = 0.5  # 길이
        marker.scale.y = 0.1  # 너비
        marker.scale.z = 0.1  # 높이
        
        # 색상 (빨간색)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # 수명
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = 500000000  # 0.5초
        
        self.pub_goal_marker.publish(marker)
        
        # 텍스트 마커 (Person ID 표시)
        text_marker = Marker()
        text_marker.header.frame_id = self.map_frame
        text_marker.header.stamp = self.get_clock().now().to_msg()
        text_marker.ns = "hand_up_goal_text"
        text_marker.id = 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        text_marker.pose.position.x = x
        text_marker.pose.position.y = y
        text_marker.pose.position.z = 0.5  # 지면 위
        
        text_marker.scale.z = 0.3
        
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        
        text_marker.text = f"Goal P{person_id}\n({x:.2f}, {y:.2f})"
        
        text_marker.lifetime.sec = 0
        text_marker.lifetime.nanosec = 500000000
        
        self.pub_goal_marker.publish(text_marker)
    
    def send_nav2_goal(self, goal_pose: PoseStamped, person_id: int):
        """Nav2에 goal 전송 (action)"""
        # Action server 대기
        if not self.nav2_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Nav2 action server not available!")
            return
        
        # NavigateToPose Goal 생성
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        
        self.get_logger().info(f"Sending navigation goal to nav2 for person {person_id}")
        
        # 목표 전송
        send_goal_future = self.nav2_client.send_goal_async(
            goal_msg,
            feedback_callback=self.nav2_feedback_callback
        )
        send_goal_future.add_done_callback(self.nav2_goal_response_callback)
        
        # 현재 타겟 설정
        self.current_target_person_id = person_id
        self.navigation_active = True
    
    def nav2_goal_response_callback(self, future):
        """Nav2 goal 응답 콜백"""
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().info("Goal rejected by nav2")
            self.navigation_active = False
            return
        
        self.get_logger().info("Goal accepted by nav2")
        
        # 결과 대기
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self.nav2_result_callback)
    
    def nav2_feedback_callback(self, feedback_msg):
        """Nav2 피드백 콜백"""
        feedback = feedback_msg.feedback
        # 필요시 피드백 로깅
        # self.get_logger().debug(f"Navigation feedback: {feedback}")
    
    def nav2_result_callback(self, future):
        """Nav2 결과 콜백"""
        result = future.result().result
        self.navigation_active = False
        self.goal_handle = None
        
        self.get_logger().info(f"Navigation completed with result: {result}")
    
    def check_distance_and_stop(self):
        """거리 모니터링하여 임계값 이하면 navigation 취소"""
        if not self.navigation_active or self.current_target_person_id is None:
            return
        
        # 현재 타겟 사람의 스켈레톤 데이터 확인
        if self.current_target_person_id not in self.skeleton_data:
            # 사람이 사라짐
            self.get_logger().warn(f"Target person {self.current_target_person_id} disappeared")
            self.cancel_navigation()
            return
        
        # 타겟 사람의 최소 거리 계산
        skeleton_info = self.skeleton_data[self.current_target_person_id]
        points = skeleton_info['points']
        distances = [np.sqrt(x**2 + y**2 + z**2) for x, y, z in points]
        min_distance = min(distances)
        
        self.get_logger().debug(f"Distance to person {self.current_target_person_id}: {min_distance:.2f}m")
        
        # 임계값 체크
        if min_distance <= self.stop_distance:
            self.get_logger().info(
                f"Reached target distance ({min_distance:.2f}m <= {self.stop_distance}m). "
                f"Canceling navigation."
            )
            self.cancel_navigation()
    
    def cancel_navigation(self):
        """Navigation 취소"""
        if self.goal_handle is not None:
            self.get_logger().info("Canceling navigation goal...")
            cancel_future = self.goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self.cancel_done_callback)
        
        self.navigation_active = False
        self.current_target_person_id = None
    
    def cancel_done_callback(self, future):
        """Navigation 취소 완료 콜백"""
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info("Navigation goal canceled successfully")
        else:
            self.get_logger().warn("Failed to cancel navigation goal")
        
        self.goal_handle = None


def main(args=None):
    rclpy.init(args=args)
    node = HandUpGoalController()
    
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
