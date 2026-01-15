#!/usr/bin/env python3
"""
손 들기 감지 평가 노드 (Hand Up Goal Evaluator) - 수정본
- 좌표계 매핑 (Optical Z -> Robot X, Optical X -> Robot -Y)
- 세션별 RMSE 계산 (GT 입력 후 10초간의 데이터만 매칭)
"""

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray
import json
from datetime import datetime
from pathlib import Path
import threading
import time


class HandUpEvaluator(Node):
    def __init__(self):
        super().__init__('hand_up_evaluator')
        
        # 파라미터 선언
        self.declare_parameter("image_topic", "/human/debug_image")
        self.declare_parameter("states_topic", "/human/states")
        self.declare_parameter("markers_topic", "/human/skeleton_markers")
        self.declare_parameter("capture_fps", 3)
        self.declare_parameter("eval_dir", "hand_up_eval_results")
        
        self.image_topic = self.get_parameter("image_topic").value
        self.states_topic = self.get_parameter("states_topic").value
        self.markers_topic = self.get_parameter("markers_topic").value
        self.capture_fps = self.get_parameter("capture_fps").value
        self.eval_path = Path(self.get_parameter("eval_dir").value)
        
        # 디렉토리 생성
        self.eval_path.mkdir(exist_ok=True)
        self.image_dir = self.eval_path / "images"
        self.image_dir.mkdir(exist_ok=True)
        
        # QoS 설정
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        
        # 구독자
        self.sub_image = self.create_subscription(Image, self.image_topic, self.on_image, qos)
        self.sub_states = self.create_subscription(String, self.states_topic, self.on_states, 10)
        self.sub_markers = self.create_subscription(MarkerArray, self.markers_topic, self.on_markers, qos)
        
        # 데이터 저장소
        self.evaluation_data = {
            "start_time": datetime.now().isoformat(),
            "trials": [],          # {gt: [x, y], preds: [[x, y], ...]} 형태의 세션 기록
            "false_positives": 0,
            "frames_captured": 0
        }
        
        # 실시간 상태 변수
        self.evaluation_active = False
        self.evaluation_start_time = None
        self.evaluation_duration = 10.0
        self.current_gt = None
        self.current_session_preds = []
        self.last_capture_time = 0
        
        # 입력 스레드
        self.input_thread = threading.Thread(target=self._input_thread, daemon=True)
        self.input_thread.start()
        
        self.get_logger().info("✅ Hand Up Evaluator Ready")
        self.get_logger().info("명령어: 'gt x y' (예: gt 3.4 0), 'fp' (오탐지), 'save' (저장), 'clear' (초기화)")

    def _input_thread(self):
        while rclpy.ok():
            try:
                user_input = input("\n[HandUpEvaluator] 명령: ").strip()
                if user_input.startswith("gt "):
                    parts = user_input.split()
                    x, y = float(parts[1]), float(parts[2])
                    self.start_evaluation_session(x, y)
                elif user_input == "fp":
                    self.evaluation_data["false_positives"] += 1
                    self.get_logger().info(f"❌ False Positive 카운트: {self.evaluation_data['false_positives']}")
                elif user_input == "save":
                    self.save_results()
                elif user_input == "clear":
                    self.evaluation_data["trials"] = []
                    self.evaluation_data["false_positives"] = 0
                    self.get_logger().info("🧹 데이터가 초기화되었습니다.")
                elif user_input in ["exit", "quit"]:
                    break
            except Exception as e:
                self.get_logger().error(f"입력 오류: {e}")

    def start_evaluation_session(self, x: float, y: float):
        """GT 입력 후 3초 대기 후 10초간 측정 시작"""
        self.current_gt = [x, y]
        self.current_session_preds = []
        
        self.get_logger().info(f"⏱️ GT({x}, {y}) 입력됨. 3초 뒤 측정을 시작합니다...")
        for i in range(3, 0, -1):
            self.get_logger().info(f">>> {i}...")
            time.sleep(1)
        
        self.evaluation_start_time = time.time()
        self.evaluation_active = True
        self.get_logger().info(f"🚀 측정 시작! (10초간)")

    def on_markers(self, msg: MarkerArray):
        """스켈레톤 마커로부터 좌표 추출 및 매핑"""
        if not self.evaluation_active or not msg.markers:
            return
        
        elapsed = time.time() - self.evaluation_start_time
        if elapsed > self.evaluation_duration:
            # 10초 종료 시 세션 데이터 저장
            self.evaluation_active = False
            self.evaluation_data["trials"].append({
                "gt": self.current_gt,
                "preds": self.current_session_preds,
                "timestamp": datetime.now().isoformat()
            })
            self.get_logger().info(f"⏹️ 측정 종료 (수집된 데이터: {len(self.current_session_preds)}개)")
            return

        try:
            joint_marker = msg.markers[0]
            l_idx, r_idx = 5, 6 # 어깨 인덱스
            if len(joint_marker.points) > r_idx:
                p_l = joint_marker.points[l_idx]
                p_r = joint_marker.points[r_idx]
                
                # Camera Optical Frame 원본
                raw_x = (p_l.x + p_r.x) / 2.0
                raw_z = (p_l.z + p_r.z) / 2.0
                
                # [좌표 변환 매핑]
                # Optical Z(정면) -> Robot X
                # Optical X(오른쪽+) -> Robot Y(왼쪽+) 이므로 부호 반전
                mapped_x = raw_z
                mapped_y = -raw_x
                
                self.current_session_preds.append([mapped_x, mapped_y])
                
                if len(self.current_session_preds) % 10 == 0:
                    self.get_logger().info(f"📍 실시간 매핑: X:{mapped_x:.2f}, Y:{mapped_y:.2f} (오차: {np.linalg.norm(np.array([mapped_x, mapped_y])-np.array(self.current_gt)):.3f}m)")
        except Exception:
            pass

    def on_image(self, msg: Image):
        """이미지 캡처 (3fps)"""
        curr = time.time()
        if curr - self.last_capture_time >= (1.0 / self.capture_fps):
            try:
                bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, -1))
                filename = self.image_dir / f"frame_{self.evaluation_data['frames_captured']:05d}.png"
                cv2.imwrite(str(filename), bgr)
                self.evaluation_data["frames_captured"] += 1
                self.last_capture_time = curr
            except Exception:
                pass

    def on_states(self, msg: String):
        pass # 필요 시 추가

    def calculate_metrics(self):
        """모든 Trial에 대한 RMSE 계산"""
        all_errors = []
        if not self.evaluation_data["trials"]:
            return {"rmse": 0.0, "mae": 0.0}
        
        for trial in self.evaluation_data["trials"]:
            gt = np.array(trial["gt"])
            preds = np.array(trial["preds"])
            if len(preds) == 0: continue
            
            # 각 예측값과 GT 사이의 유클리드 거리 계산
            dists = np.linalg.norm(preds - gt, axis=1)
            all_errors.extend(dists.tolist())
            
        if not all_errors:
            return {"rmse": 0.0, "mae": 0.0}
            
        errors_np = np.array(all_errors)
        rmse = np.sqrt(np.mean(errors_np**2))
        mae = np.mean(errors_np)
        return {"rmse": float(rmse), "mae": float(mae)}

    def save_results(self):
        metrics = self.calculate_metrics()
        final_results = {
            "summary": metrics,
            "false_positives": self.evaluation_data["false_positives"],
            "total_trials": len(self.evaluation_data["trials"]),
            "details": self.evaluation_data["trials"]
        }
        
        filename = self.eval_path / f"eval_{datetime.now().strftime('%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(final_results, f, indent=2)
            
        self.get_logger().info(f"\n" + "="*50)
        self.get_logger().info(f"📊 최종 평가 결과 ({len(self.evaluation_data['trials'])} 세션)")
        self.get_logger().info(f"  - 좌표 RMSE: {metrics['rmse']:.4f} m")
        self.get_logger().info(f"  - 좌표 MAE : {metrics['mae']:.4f} m")
        self.get_logger().info(f"  - 거짓 양성: {final_results['false_positives']} 회")
        self.get_logger().info(f"  - 결과 저장: {filename}")
        self.get_logger().info("="*50 + "\n")


def main(args=None):
    rclpy.init(args=args)
    node = HandUpEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()