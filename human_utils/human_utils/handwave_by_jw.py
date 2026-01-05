#!/home/nvidia/vision_ws/human_venv/bin/python

import time, math
from collections import deque
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
from ultralytics import YOLO

def angle(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]], dtype=float)
    bc = np.array([c[0]-b[0], c[1]-b[1]], dtype=float)
    nba = ba / (np.linalg.norm(ba) + 1e-6)
    nbc = bc / (np.linalg.norm(bc) + 1e-6)
    cosang = float(np.clip(np.dot(nba, nbc), -1.0, 1.0))
    return math.degrees(math.acos(cosang))

def get_yolo_keypoints(results):
    """YOLO Pose 결과에서 keypoint 추출
    COCO format 17개: 0=nose, 5=left_shoulder, 6=right_shoulder,
                      7=left_elbow, 8=right_elbow, 9=left_wrist, 10=right_wrist,
                      11=left_hip, 12=right_hip
    """
    if len(results) == 0 or results[0].keypoints is None:
        return {}
    
    kp = results[0].keypoints.data[0].cpu().numpy()  # (17, 3) [x, y, conf]
    h, w = results[0].orig_shape
    
    pose = {}
    idx_map = {
        "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10,
        "left_hip": 11, "right_hip": 12,
    }
    
    for name, idx in idx_map.items():
        if idx < len(kp) and kp[idx][2] > 0.3:  # confidence > 0.3
            pose[name] = (int(kp[idx][0]), int(kp[idx][1]), float(kp[idx][2]))
    
    return pose

def is_hand_up(side, pose, shoulder_y, torso_px,
               min_arm_deg=150, shoulder_margin_px=0.0,
               ignore_height=True, use_relaxed=True,
               min_arm_deg_relaxed=70, elbow_margin_px=0.0):
    wrist = pose[f"{side}_wrist"]
    elbow = pose[f"{side}_elbow"]
    shoulder = pose[f"{side}_shoulder"]
    
    ang = angle((shoulder[0], shoulder[1]),
                (elbow[0], elbow[1]),
                (wrist[0], wrist[1]))
    
    strict_height_ok = True if ignore_height else (wrist[1] <= (shoulder_y + float(shoulder_margin_px)))
    strict_ok = strict_height_ok and (ang >= float(min_arm_deg))
    
    relaxed_ok = False
    if use_relaxed:
        relaxed_height_ok = (wrist[1] <= (elbow[1] - float(elbow_margin_px)))
        relaxed_ok = relaxed_height_ok and (ang >= float(min_arm_deg_relaxed))
    
    return strict_ok or relaxed_ok

def detect_waving_lr_only(
    wrist_queue, frame_width, scale_px=None,
    min_secs=1.2, min_zero_cross=4,
    min_amp_ratio=0.05, amp_per_scale=0.6,
    min_amp_px_floor=5.0, horiz_dom_ratio=1.5,
    base_speed_pxps=1.0, ref_scale_px=120.0,
    min_speed_px_floor=0.5
):
    if len(wrist_queue) < 10:
        return False
    
    now = wrist_queue[-1][2]
    window = [(x, y, t) for (x, y, t) in wrist_queue if now - t <= min_secs]
    if len(window) < 10:
        return False
    
    duration = window[-1][2] - window[0][2]
    if duration < (min_secs - 0.05):
        return False
    
    xs = np.array([x for (x, _, _) in window], dtype=float)
    ys = np.array([y for (_, y, _) in window], dtype=float)
    ts = np.array([t for (*_, t) in window], dtype=float)
    
    x_detrend = xs - xs.mean()
    zero_cross = int(np.sum((x_detrend[:-1] * x_detrend[1:]) < 0))
    if zero_cross < min_zero_cross:
        return False
    
    amp_px = float(np.max(xs) - np.min(xs))
    amp_thresh_frame = float(min_amp_ratio) * float(max(frame_width, 1.0))
    amp_thresh_scale = amp_per_scale * float(scale_px) if scale_px else float('inf')
    amp_thresh = max(min_amp_px_floor, min(amp_thresh_frame, amp_thresh_scale))
    if amp_px < amp_thresh:
        return False
    
    std_x, std_y = float(np.std(xs)), float(np.std(ys))
    if std_x < horiz_dom_ratio * max(std_y, 1e-6):
        return False
    
    dt = np.diff(ts)
    dx = np.diff(xs)
    valid = dt > 1e-6
    if not np.any(valid):
        return False
    
    mean_speed = float(np.mean(np.abs(dx[valid] / dt[valid])))
    speed_thresh = base_speed_pxps * (float(scale_px) / ref_scale_px) if scale_px else base_speed_pxps
    speed_thresh = max(min_speed_px_floor, speed_thresh)
    if mean_speed < speed_thresh:
        return False
    
    return True

class YoloPoseGestureNode(Node):
    def __init__(self):
        super().__init__("yolo_pose_gesture_node")

        # 파라미터
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("show_preview", False)
        self.declare_parameter("mirror_view", False)
        
        # YOLO 설정
        self.declare_parameter("yolo_model", "yolov8s-pose.pt")  # small 모델
        self.declare_parameter("yolo_conf", 0.5)
        self.declare_parameter("yolo_device", "cuda")  # 'cuda' 또는 'cpu'

        # Waving/HandUp 파라미터 (기존과 동일)
        self.declare_parameter("waving_min_zero_cross", 2)
        self.declare_parameter("waving_min_amp_ratio", 0.05)
        self.declare_parameter("waving_horiz_dom_ratio", 1.5)
        self.declare_parameter("waving_min_mean_speed_px", 0.5)
        self.declare_parameter("waving_min_secs", 1.5)
        self.declare_parameter("waving_stable_need", 15)
        self.declare_parameter("waving_decay", 5)
        
        self.declare_parameter("handup_min_arm_deg", 100)
        self.declare_parameter("handup_min_arm_deg_relaxed", 35)
        self.declare_parameter("handup_shoulder_margin_ratio", 0.15)
        self.declare_parameter("handup_elbow_margin_ratio", 0.05)
        self.declare_parameter("handup_ignore_height", False)
        self.declare_parameter("handup_use_relaxed", True)
        
        self.declare_parameter("state_min_hold_sec", 1.5)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        caminfo_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value

        self.show_preview = bool(self.get_parameter("show_preview").value)
        self.mirror_view = bool(self.get_parameter("mirror_view").value)

        if not image_topic.startswith('/'): image_topic = '/' + image_topic
        if not caminfo_topic.startswith('/'): caminfo_topic = '/' + caminfo_topic
        if not depth_topic.startswith('/'): depth_topic = '/' + depth_topic

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(Image, image_topic, self.on_image, qos)
        self.sub_info = self.create_subscription(CameraInfo, caminfo_topic, self.on_info, qos)
        self.sub_depth = self.create_subscription(Image, depth_topic, self.on_depth, qos)

        self.pub_gesture = self.create_publisher(String, "/gesture", 10)
        self.pub_dbg = self.create_publisher(Image, "/gesture/debug_image", 10)
        self.pub_distance = self.create_publisher(Float32, "/gesture/hand_distance", 10)

        # YOLO 모델 로드
        model_name = self.get_parameter("yolo_model").get_parameter_value().string_value
        device = self.get_parameter("yolo_device").get_parameter_value().string_value
        
        self.get_logger().info(f"Loading YOLO model: {model_name} on {device}")
        self.model = YOLO(model_name)
        if device == 'cuda':
            self.model.to('cuda')

        self.yolo_conf = float(self.get_parameter("yolo_conf").value)

        self.left_wrist_hist = deque(maxlen=60)
        self.right_wrist_hist = deque(maxlen=60)
        self.waving_stable_ctr = 0
        self.last_time = time.time()
        self.fps_est = 10.0
        self.caminfo = None

        self.raw_gesture = ""
        self.confirmed_gesture = ""
        self.last_state_change = time.time()

        self.depth_image = None
        self.depth_dtype = None

        self.get_logger().info(f"Subscribed to: {image_topic}, {caminfo_topic}, {depth_topic}")

    def on_info(self, msg: CameraInfo):
        self.caminfo = msg

    def on_depth(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge depth convert failed: {e}")
            return
        if depth is None:
            return
        self.depth_image = depth
        self.depth_dtype = depth.dtype

    def estimate_distance_from_pose(self, pose_xy):
        if self.depth_image is None:
            return None

        depth = self.depth_image
        h, w = depth.shape[:2]

        if "left_shoulder" in pose_xy and "right_shoulder" in pose_xy:
            x = (pose_xy["left_shoulder"][0] + pose_xy["right_shoulder"][0]) // 2
            y = (pose_xy["left_shoulder"][1] + pose_xy["right_shoulder"][1]) // 2
        elif "left_shoulder" in pose_xy:
            x, y = pose_xy["left_shoulder"][:2]
        elif "right_shoulder" in pose_xy:
            x, y = pose_xy["right_shoulder"][:2]
        elif "left_hip" in pose_xy and "right_hip" in pose_xy:
            x = (pose_xy["left_hip"][0] + pose_xy["right_hip"][0]) // 2
            y = (pose_xy["left_hip"][1] + pose_xy["right_hip"][1]) // 2
        elif "left_wrist" in pose_xy:
            x, y = pose_xy["left_wrist"][:2]
        elif "right_wrist" in pose_xy:
            x, y = pose_xy["right_wrist"][:2]
        else:
            return None

        x = int(np.clip(x, 0, w-1))
        y = int(np.clip(y, 0, h-1))

        y1 = max(0, y-2); y2 = min(h, y+3)
        x1 = max(0, x-2); x2 = min(w, x+3)
        patch = depth[y1:y2, x1:x2]

        patch_flat = patch.reshape(-1)
        valid = patch_flat[(patch_flat > 0) & np.isfinite(patch_flat)]
        if valid.size == 0:
            return None

        d_raw = float(np.median(valid))

        if self.depth_dtype == np.uint16:
            dist_m = d_raw / 1000.0
        else:
            dist_m = d_raw

        if dist_m <= 0 or dist_m > 20.0:
            return None

        return dist_m

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"cv_bridge convert failed: {e}")
            return

        now = time.time()
        dt = now - self.last_time
        self.last_time = now
        if dt > 0:
            self.fps_est = 0.9 * self.fps_est + 0.1 * (1.0 / dt)

        # YOLO 추론 (GPU에서 실행)
        results = self.model(bgr, conf=self.yolo_conf, verbose=False)
        pose_xy = get_yolo_keypoints(results)

        frame = bgr.copy()
        h, w = frame.shape[:2]

        if "left_wrist" in pose_xy:
            lw = pose_xy["left_wrist"]
            self.left_wrist_hist.append((lw[0], lw[1], now))
        if "right_wrist" in pose_xy:
            rw = pose_xy["right_wrist"]
            self.right_wrist_hist.append((rw[0], rw[1], now))

        l_scale = r_scale = None
        if all(k in pose_xy for k in ["left_wrist", "left_elbow"]):
            lw = pose_xy["left_wrist"]; le = pose_xy["left_elbow"]
            l_scale = float(np.hypot(lw[0]-le[0], lw[1]-le[1]))
        if all(k in pose_xy for k in ["right_wrist", "right_elbow"]):
            rw = pose_xy["right_wrist"]; re = pose_xy["right_elbow"]
            r_scale = float(np.hypot(rw[0]-re[0], rw[1]-re[1]))

        handup_left = handup_right = False
        if "left_shoulder" in pose_xy and "right_shoulder" in pose_xy:
            shoulder_y = int((pose_xy["left_shoulder"][1] + pose_xy["right_shoulder"][1]) / 2)
        else:
            shoulder_y = int(h * 0.15)

        if all(k in pose_xy for k in ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]):
            hip_y = int((pose_xy["left_hip"][1] + pose_xy["right_hip"][1]) / 2)
            torso_px = abs(hip_y - shoulder_y)
        else:
            torso_px = h * 0.25

        margin_ratio = float(self.get_parameter("handup_shoulder_margin_ratio").value)
        shoulder_margin_px = max(8.0, margin_ratio * float(torso_px))
        elbow_margin_ratio = float(self.get_parameter("handup_elbow_margin_ratio").value)
        elbow_margin_px = max(5.0, elbow_margin_ratio * float(torso_px))

        ignore_height = bool(self.get_parameter("handup_ignore_height").value)
        use_relaxed = bool(self.get_parameter("handup_use_relaxed").value)
        min_arm_deg_strict = float(self.get_parameter("handup_min_arm_deg").value)
        min_arm_deg_relaxed = float(self.get_parameter("handup_min_arm_deg_relaxed").value)

        if all(k in pose_xy for k in ["left_wrist", "left_elbow", "left_shoulder"]):
            if is_hand_up("left", pose_xy, shoulder_y, torso_px,
                          min_arm_deg=min_arm_deg_strict,
                          shoulder_margin_px=shoulder_margin_px,
                          ignore_height=ignore_height,
                          use_relaxed=use_relaxed,
                          min_arm_deg_relaxed=min_arm_deg_relaxed,
                          elbow_margin_px=elbow_margin_px):
                handup_left = True
        
        if all(k in pose_xy for k in ["right_wrist", "right_elbow", "right_shoulder"]):
            if is_hand_up("right", pose_xy, shoulder_y, torso_px,
                          min_arm_deg=min_arm_deg_strict,
                          shoulder_margin_px=shoulder_margin_px,
                          ignore_height=ignore_height,
                          use_relaxed=use_relaxed,
                          min_arm_deg_relaxed=min_arm_deg_relaxed,
                          elbow_margin_px=elbow_margin_px):
                handup_right = True

        # Waving 감지
        if any([
            detect_waving_lr_only(
                self.left_wrist_hist, frame_width=w, scale_px=l_scale,
                min_secs=float(self.get_parameter("waving_min_secs").value),
                min_zero_cross=int(self.get_parameter("waving_min_zero_cross").value),
                min_amp_ratio=float(self.get_parameter("waving_min_amp_ratio").value),
                horiz_dom_ratio=float(self.get_parameter("waving_horiz_dom_ratio").value),
                base_speed_pxps=float(self.get_parameter("waving_min_mean_speed_px").value),
            ),
            detect_waving_lr_only(
                self.right_wrist_hist, frame_width=w, scale_px=r_scale,
                min_secs=float(self.get_parameter("waving_min_secs").value),
                min_zero_cross=int(self.get_parameter("waving_min_zero_cross").value),
                min_amp_ratio=float(self.get_parameter("waving_min_amp_ratio").value),
                horiz_dom_ratio=float(self.get_parameter("waving_horiz_dom_ratio").value),
                base_speed_pxps=float(self.get_parameter("waving_min_mean_speed_px").value),
            )
        ]):
            self.waving_stable_ctr += 1
        else:
            self.waving_stable_ctr = max(0, self.waving_stable_ctr - int(self.get_parameter("waving_decay").value))

        if self.waving_stable_ctr >= int(self.get_parameter("waving_stable_need").value):
            raw = "WAVING"
        else:
            if handup_left and handup_right:
                raw = "HAND UP BOTH"
            elif handup_left:
                raw = "HAND UP LEFT"
            elif handup_right:
                raw = "HAND UP RIGHT"
            else:
                raw = ""

        # 상태 홀드 로직
        hold_sec = float(self.get_parameter("state_min_hold_sec").value)
        if raw != self.raw_gesture:
            self.raw_gesture = raw
            self.last_state_change = now

        if (now - self.last_state_change) >= hold_sec:
            if self.confirmed_gesture != self.raw_gesture:
                self.confirmed_gesture = self.raw_gesture

        # 퍼블리시
        if self.confirmed_gesture:
            self.pub_gesture.publish(String(data=self.confirmed_gesture))

        # 거리 추정
        dist_m = None
        if self.confirmed_gesture and len(pose_xy) > 0:
            dist_m = self.estimate_distance_from_pose(pose_xy)
            if dist_m is not None:
                self.pub_distance.publish(Float32(data=float(dist_m)))

        # 디버그 시각화 (YOLO 내장 플롯 사용)
        annotated_frame = results[0].plot()
        
        if self.confirmed_gesture:
            cv2.putText(annotated_frame, self.confirmed_gesture, (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.putText(annotated_frame, f"FPS: {self.fps_est:.1f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if dist_m is not None:
            cv2.putText(annotated_frame, f"Dist: {dist_m:.2f} m", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        show_frame = cv2.flip(annotated_frame, 1) if self.mirror_view else annotated_frame

        out_msg = self.bridge.cv2_to_imgmsg(show_frame, encoding="bgr8")
        out_msg.header = msg.header
        self.pub_dbg.publish(out_msg)

        if self.show_preview:
            cv2.imshow("YOLO Gesture Debug", show_frame)
            cv2.waitKey(1)

def main():
    rclpy.init()
    node = YoloPoseGestureNode()
    node.get_logger().info("=== Starting YOLO Pose Gesture Node ===")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:
                pass

if __name__ == "__main__":
    main()