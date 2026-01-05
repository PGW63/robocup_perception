#!/home/nvidia/vision_ws/src/detection_venv/bin/python
"""
GroundedSAM2 Detection & Tracking Node for ROS2 Humble

기능:
- /detection_node/use_open_set 토픽으로 YOLO와 연동 제어
- use_open_set=True일 때만 동작
- 특정 객체를 찾을 때만 사용 (String 토픽으로 객체 지정)
- GroundingDINO로 bbox 탐지 후 SAM2로 마스크 생성 & 트래킹
- 2초 내 못찾으면 NOT_FOUND 퍼블리시 후 YOLO 모드로 복귀
- vision_msgs/Detection2DArray로 결과 퍼블리시

상태 머신:
- IDLE: 비활성화 상태
- SEARCHING: GroundingDINO로 객체 탐색 중 (2초 타임아웃)
- TRACKING: SAM2로 객체 트래킹 중
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2, PointField
from std_msgs.msg import String, Bool, Header
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import message_filters
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
import time
import os
import sys
import struct
import json

from rcl_interfaces.msg import SetParametersResult

# Grounded-SAM-2 패키지 경로 추가
GROUNDED_SAM2_PATH = os.path.expanduser("~/vision_ws/src/Grounded-SAM-2")
if GROUNDED_SAM2_PATH not in sys.path:
    sys.path.insert(0, GROUNDED_SAM2_PATH)

# GroundingDINO (로컬 패키지)
try:
    from grounding_dino.groundingdino.util.inference import load_model as load_gdino_model, predict as gdino_predict
    import grounding_dino.groundingdino.datasets.transforms as T
    GDINO_AVAILABLE = True
except ImportError as e:
    GDINO_AVAILABLE = False
    print(f"[WARN] GroundingDINO not available: {e}")

# SAM2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError as e:
    SAM2_AVAILABLE = False
    print(f"[WARN] SAM2 not available: {e}")


class GroundedSAM2Node(Node):
    """
    GroundedSAM2 노드: Open-set detection & tracking
    
    입력 프롬프트는 GroundingDINO 형식 그대로 사용:
    - "apple." 또는 "red apple. green bottle." 형태
    - 마지막에 점(.)이 없으면 자동 추가
    """
    
    # 상태 상수
    STATE_IDLE = "IDLE"
    STATE_SEARCHING = "SEARCHING"
    STATE_TRACKING = "TRACKING"
    
    def __init__(self):
        super().__init__('grounded_sam2_node')
        
        # ==================== Parameters ====================
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('image_type', 'compressed')  # 'raw' or 'compressed'
        self.declare_parameter('search_timeout', 2.0)  # 탐색 타임아웃 (초)
        self.declare_parameter('box_threshold', 0.35)
        self.declare_parameter('text_threshold', 0.25)
        # 모델 경로 (Grounded-SAM-2 패키지 기준 상대 경로)
        self.declare_parameter('gdino_config', 'grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py')
        self.declare_parameter('gdino_checkpoint', 'gdino_checkpoints/groundingdino_swint_ogc.pth')
        self.declare_parameter('sam2_config', 'configs/sam2.1/sam2.1_hiera_b+.yaml')  # SAM2 Hydra config 상대 경로
        self.declare_parameter('sam2_checkpoint', 'checkpoints/sam2.1_hiera_base_plus.pt')
        
        # PointCloud 관련 파라미터
        self.declare_parameter('filtered_depth_topic', '/detection_node/filtered_depth_image')
        self.declare_parameter('camera_info_topic', '/camera/camera/aligned_depth_to_color/camera_info')
        self.declare_parameter('enable_pointcloud', True)  # 포인트클라우드 생성 활성화
        self.declare_parameter('min_depth', 0.1)  # 최소 depth (m)
        self.declare_parameter('max_depth', 2.0)  # 최대 depth (m)
        self.declare_parameter('depth_scale', 0.001)  # depth 스케일 (mm -> m)
        
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.image_type = self.get_parameter('image_type').get_parameter_value().string_value
        self.search_timeout = self.get_parameter('search_timeout').get_parameter_value().double_value
        self.box_threshold = self.get_parameter('box_threshold').get_parameter_value().double_value
        self.text_threshold = self.get_parameter('text_threshold').get_parameter_value().double_value
        
        # 모델 경로 설정
        self.gdino_config = os.path.join(GROUNDED_SAM2_PATH, 
            self.get_parameter('gdino_config').get_parameter_value().string_value)
        self.gdino_checkpoint = os.path.join(GROUNDED_SAM2_PATH,
            self.get_parameter('gdino_checkpoint').get_parameter_value().string_value)
        self.sam2_config = self.get_parameter('sam2_config').get_parameter_value().string_value
        self.sam2_checkpoint = os.path.join(GROUNDED_SAM2_PATH,
            self.get_parameter('sam2_checkpoint').get_parameter_value().string_value)
        
        # ==================== State ====================
        self.state = self.STATE_IDLE
        self.text_prompt = ""  # GroundingDINO 프롬프트 (예: "red apple. green bottle.")
        self.search_start_time = None
        self.bridge = CvBridge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # use_open_set=True면 GroundedSAM2 활성화, False면 비활성화 (YOLO가 동작)
        self.use_open_set = False
        
        # Tracking state
        self.tracking_masks = None
        self.tracking_boxes = None
        self.tracking_labels = []
        self.tracking_scores = []
        self.current_segmentation_masks = []  # 현재 프레임의 2D 마스크들
        
        # Image transform for GroundingDINO
        self.gdino_transform = None
        
        # PointCloud 관련 상태
        self.filtered_depth_topic = self.get_parameter('filtered_depth_topic').get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        self.enable_pointcloud = self.get_parameter('enable_pointcloud').get_parameter_value().bool_value
        self.min_depth = self.get_parameter('min_depth').get_parameter_value().double_value
        self.max_depth = self.get_parameter('max_depth').get_parameter_value().double_value
        self.depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        
        # Camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_frame = None
        
        # 최신 필터링된 depth 이미지
        self.latest_filtered_depth = None
        self.latest_depth_header = None
        
        # ==================== Publishers (공통 토픽) ====================
        self.status_pub = self.create_publisher(String, '/detection_node/status', 10)
        self.debug_image_pub = self.create_publisher(Image, '/detection_node/debug_image', 10)
        self.detection_pub = self.create_publisher(Detection2DArray, '/detection_node/detections', 10)
        self.use_open_set_pub = self.create_publisher(Bool, '/detection_node/use_open_set', 10)
        
        # PointCloud publishers
        self.pointcloud_pub = self.create_publisher(
            PointCloud2, '/detection_node/segmented_pointcloud', 10)
        self.object_info_pub = self.create_publisher(
            String, '/detection_node/object_info', 10)
        
        # ==================== Subscribers ====================
        # 모드 제어용 (YOLO와 공유)
        self.mode_sub = self.create_subscription(
            Bool, '/detection_node/use_open_set', self.mode_callback, 10)
        
        # 탐색할 객체 지정 (이 토픽으로 메시지가 오면 탐색 시작)
        # 입력 예: "green apple. coke cola." 또는 "apple"
        self.search_sub = self.create_subscription(
            String, '/detection_node/search', self.search_callback, 10)
        
        # 탐색/트래킹 중지
        self.stop_sub = self.create_subscription(
            Bool, '/detection_node/stop', self.stop_callback, 10)
        
        # Camera subscription
        self.subscription = None
        self._create_image_subscription()
        
        # Depth & Camera Info subscription (PointCloud 생성용)
        if self.enable_pointcloud:
            self.depth_sub = self.create_subscription(
                Image, self.filtered_depth_topic,
                self.filtered_depth_callback, 10)
            
            self.camera_info_sub = self.create_subscription(
                CameraInfo, self.camera_info_topic,
                self.camera_info_callback, 10)
        
        # ==================== Model Loading ====================
        self._load_models()
        
        # Parameter callback
        self._cb_handle = self.add_on_set_parameters_callback(self.on_param_change)
        
        # Timer for timeout check
        self.timeout_timer = self.create_timer(0.1, self.check_timeout)
        
        self._print_init_info()
    
    def _print_init_info(self):
        self.get_logger().info("=" * 60)
        self.get_logger().info("GroundedSAM2 Node initialized")
        self.get_logger().info(f"  - Image topic: {self.image_topic} ({self.image_type})")
        self.get_logger().info(f"  - Device: {self.device}")
        self.get_logger().info(f"  - Search timeout: {self.search_timeout}s")
        self.get_logger().info(f"  - Box threshold: {self.box_threshold}")
        self.get_logger().info(f"  - Text threshold: {self.text_threshold}")
        self.get_logger().info(f"  - GroundingDINO: {'Loaded' if self.gdino_model else 'Not available'}")
        self.get_logger().info(f"  - SAM2: {'Loaded' if self.sam2_predictor else 'Not available'}")
        self.get_logger().info(f"  - Mode control: /detection_node/use_open_set")
        self.get_logger().info(f"      True  -> GroundedSAM2 active")
        self.get_logger().info(f"      False -> YOLO active (default)")
        self.get_logger().info(f"  - Search input: /detection_node/search")
        self.get_logger().info(f"      예: 'green apple. coke cola.' 또는 'apple'")
        self.get_logger().info(f"  - Output:")
        self.get_logger().info(f"      /detection_node/detections")
        self.get_logger().info(f"      /detection_node/debug_image")
        if self.enable_pointcloud:
            self.get_logger().info(f"  - PointCloud enabled:")
            self.get_logger().info(f"      Input depth: {self.filtered_depth_topic}")
            self.get_logger().info(f"      Camera info: {self.camera_info_topic}")
            self.get_logger().info(f"      Output: /detection_node/segmented_pointcloud")
            self.get_logger().info(f"      Depth range: {self.min_depth}m ~ {self.max_depth}m")
        self.get_logger().info("=" * 60)
    
    def _load_models(self):
        """GroundingDINO와 SAM2 모델 로드"""
        self.gdino_model = None
        self.sam2_predictor = None
        
        # CUDA 설정 (GroundingDINO가 bfloat16을 지원하지 않으므로 autocast 사용 안 함)
        if self.device == "cuda":
            # torch.autocast는 GroundingDINO의 ms_deform_attn과 호환 안 됨
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        
        # 작업 디렉토리를 Grounded-SAM-2로 변경 (SAM2 Hydra config 로드를 위해)
        original_cwd = os.getcwd()
        os.chdir(GROUNDED_SAM2_PATH)
        
        try:
            # GroundingDINO 로드 (로컬 패키지)
            if GDINO_AVAILABLE:
                try:
                    if os.path.exists(self.gdino_config) and os.path.exists(self.gdino_checkpoint):
                        self.get_logger().info(f"Loading GroundingDINO...")
                        self.get_logger().info(f"  Config: {self.gdino_config}")
                        self.get_logger().info(f"  Checkpoint: {self.gdino_checkpoint}")
                        
                        self.gdino_model = load_gdino_model(
                            model_config_path=self.gdino_config,
                            model_checkpoint_path=self.gdino_checkpoint,
                            device=self.device
                        )
                        
                        # Image transform 설정
                        self.gdino_transform = T.Compose([
                            T.RandomResize([800], max_size=1333),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])
                        
                        self.get_logger().info("GroundingDINO loaded successfully")
                    else:
                        self.get_logger().error(f"GroundingDINO files not found")
                        self.get_logger().error(f"  Config exists: {os.path.exists(self.gdino_config)}")
                        self.get_logger().error(f"  Checkpoint exists: {os.path.exists(self.gdino_checkpoint)}")
                except Exception as e:
                    self.get_logger().error(f"Failed to load GroundingDINO: {e}")
            
            # SAM2 로드 (상대 경로 사용 - cwd가 GROUNDED_SAM2_PATH)
            if SAM2_AVAILABLE:
                try:
                    if os.path.exists(self.sam2_checkpoint):
                        self.get_logger().info(f"Loading SAM2...")
                        self.get_logger().info(f"  Config: {self.sam2_config} (relative to {GROUNDED_SAM2_PATH})")
                        self.get_logger().info(f"  Checkpoint: {self.sam2_checkpoint}")
                        
                        sam2_model = build_sam2(self.sam2_config, self.sam2_checkpoint, device=self.device)
                        self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                        self.get_logger().info("SAM2 loaded successfully")
                    else:
                        self.get_logger().error(f"SAM2 checkpoint not found: {self.sam2_checkpoint}")
                except Exception as e:
                    self.get_logger().error(f"Failed to load SAM2: {e}")
                    import traceback
                    traceback.print_exc()
        finally:
            # 원래 작업 디렉토리로 복원
            os.chdir(original_cwd)
    
    def _create_image_subscription(self):
        """이미지 토픽 구독 생성"""
        if self.subscription is not None:
            self.destroy_subscription(self.subscription)
        
        if self.image_type == 'raw':
            self.subscription = self.create_subscription(
                Image, self.image_topic, self.image_raw_callback, 10)
        else:
            self.subscription = self.create_subscription(
                CompressedImage, self.image_topic, self.image_compressed_callback, 10)
    
    # ==================== Depth & PointCloud Callbacks ====================
    
    def camera_info_callback(self, msg: CameraInfo):
        """카메라 내부 파라미터 획득"""
        if self.fx is None:
            K = np.array(msg.k).reshape(3, 3)
            self.fx = K[0, 0]
            self.fy = K[1, 1]
            self.cx = K[0, 2]
            self.cy = K[1, 2]
            self.camera_frame = msg.header.frame_id
            
            self.get_logger().info(
                f"Camera intrinsics received: fx={self.fx:.1f}, fy={self.fy:.1f}, "
                f"cx={self.cx:.1f}, cy={self.cy:.1f}, frame={self.camera_frame}")
    
    def filtered_depth_callback(self, msg: Image):
        """필터링된 depth 이미지 수신"""
        try:
            self.latest_filtered_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_depth_header = msg.header
        except Exception as e:
            self.get_logger().warn(f"Failed to convert depth image: {e}")
    
    def _preprocess_prompt(self, text: str) -> str:
        """
        GroundingDINO 프롬프트 전처리
        - 소문자 변환
        - 마지막에 점(.)이 없으면 추가
        
        입력: "Green Apple" 또는 "green apple. Coke Cola"
        출력: "green apple." 또는 "green apple. coke cola."
        """
        result = text.lower().strip()
        if not result.endswith("."):
            result += "."
        return result
    
    # ==================== Callbacks ====================
    
    def mode_callback(self, msg: Bool):
        """모드 전환: use_open_set=True면 GroundedSAM2, False면 YOLO"""
        self.use_open_set = msg.data
        if self.use_open_set:
            self.get_logger().info("Mode: GroundedSAM2 active")
        else:
            self.get_logger().info("Mode: YOLO active (GroundedSAM2 paused)")
            # YOLO 모드로 전환시 트래킹 중지
            if self.state != self.STATE_IDLE:
                self._reset_state()
                self._publish_status("STOPPED")
    
    def search_callback(self, msg: String):
        """
        탐색 시작 콜백
        
        입력 형식 (GroundingDINO 프롬프트 그대로):
        - "apple"
        - "green apple"  
        - "green apple. coke cola."
        - "red apple. blue bottle. yellow cup."
        """
        raw_prompt = msg.data.strip()
        if not raw_prompt:
            self.get_logger().warn("Empty search prompt received")
            return
        
        # 프롬프트 전처리 (소문자 + 마지막 점 추가)
        self.text_prompt = self._preprocess_prompt(raw_prompt)
        
        # search 메시지가 들어오면 자동으로 GroundedSAM2 모드로 전환
        self.use_open_set = True
        self.state = self.STATE_SEARCHING
        self.search_start_time = time.time()
        self._reset_tracking_state()
        
        self.get_logger().info(f"Starting search...")
        self.get_logger().info(f"  Raw input: '{raw_prompt}'")
        self.get_logger().info(f"  Processed prompt: '{self.text_prompt}'")
        
        self._publish_status("SEARCHING")
    
    def stop_callback(self, msg: Bool):
        """트래킹/탐색 중지"""
        if msg.data:
            self.get_logger().info("Stopping search/tracking")
            self._reset_state()
            self._publish_status("STOPPED")
            # YOLO 모드로 복귀
            self.use_open_set = False
            self.use_open_set_pub.publish(Bool(data=self.use_open_set))
    
    def image_raw_callback(self, msg: Image):
        if not self.use_open_set or self.state == self.STATE_IDLE:
            return
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.process(cv_image, msg.header)
    
    def image_compressed_callback(self, msg: CompressedImage):
        if not self.use_open_set or self.state == self.STATE_IDLE:
            return
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        self.process(cv_image, msg.header)
    
    def check_timeout(self):
        """탐색 타임아웃 체크"""
        if self.state == self.STATE_SEARCHING and self.search_start_time:
            elapsed = time.time() - self.search_start_time
            if elapsed > self.search_timeout:
                self.get_logger().warn(f"Search timeout ({self.search_timeout}s) - object not found")
                self._publish_status("NOT_FOUND")
                self._reset_state()
                # 타임아웃 시 자동으로 YOLO 모드로 복귀
                self.use_open_set = False
                self.use_open_set_pub.publish(Bool(data=self.use_open_set))
                self.get_logger().info("Auto switching back to YOLO mode")
    
    def on_param_change(self, params):
        for p in params:
            if p.name == 'box_threshold':
                self.box_threshold = p.value
                self.get_logger().info(f"box_threshold -> {self.box_threshold}")
            elif p.name == 'text_threshold':
                self.text_threshold = p.value
                self.get_logger().info(f"text_threshold -> {self.text_threshold}")
            elif p.name == 'search_timeout':
                self.search_timeout = p.value
                self.get_logger().info(f"search_timeout -> {self.search_timeout}")
        return SetParametersResult(successful=True)
    
    # ==================== State Management ====================
    
    def _reset_state(self):
        """상태 초기화"""
        self.state = self.STATE_IDLE
        self.text_prompt = ""
        self.search_start_time = None
        self._reset_tracking_state()
    
    def _reset_tracking_state(self):
        """트래킹 상태만 초기화"""
        self.tracking_masks = None
        self.tracking_boxes = None
        self.tracking_labels = []
        self.tracking_scores = []
    
    def _publish_status(self, status: str):
        """상태 퍼블리시"""
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
    
    # ==================== Processing ====================
    
    def process(self, img_bgr: np.ndarray, header):
        """메인 처리 함수"""
        if self.state == self.STATE_SEARCHING:
            self._process_search(img_bgr, header)
        elif self.state == self.STATE_TRACKING:
            self._process_tracking(img_bgr, header)
    
    def _process_search(self, img_bgr: np.ndarray, header):
        """GroundingDINO로 객체 탐색"""
        if self.gdino_model is None:
            self.get_logger().error("GroundingDINO not loaded")
            return
        
        img_copy = img_bgr.copy()
        
        if not self.text_prompt:
            self._publish_empty_detection(img_copy, header)
            return
        
        H, W = img_bgr.shape[:2]
        
        try:
            # OpenCV BGR -> PIL RGB -> Transform
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(img_rgb)
            img_transformed, _ = self.gdino_transform(pil_img, None)
            
            # GroundingDINO 추론
            boxes, scores, phrases = gdino_predict(
                model=self.gdino_model,
                image=img_transformed,
                caption=self.text_prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )
            
            # 결과 확인
            if len(boxes) > 0:
                # cxcywh -> xyxy 변환 및 이미지 크기에 맞게 스케일
                boxes_scaled = boxes * torch.Tensor([W, H, W, H])
                from torchvision.ops import box_convert
                boxes_xyxy = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
                scores_np = scores.numpy()
                
                self.get_logger().info(f"Found {len(boxes_xyxy)} objects: {phrases}")
                
                # SAM2로 마스크 생성 및 트래킹 전환
                if self.sam2_predictor is not None:
                    self._init_tracking_with_sam2(img_bgr, boxes_xyxy, phrases, scores_np)
                else:
                    self._init_tracking_bbox_only(boxes_xyxy, phrases, scores_np)
                
                # 탐색 성공 - 트래킹 모드로 전환
                self.state = self.STATE_TRACKING
                self._publish_status("FOUND")
                
                # 시각화 및 퍼블리시
                self._visualize_and_publish(img_copy, header)
                return
            
            # 못 찾음 - 계속 탐색
            self._publish_empty_detection(img_copy, header)
                
        except Exception as e:
            self.get_logger().error(f"GroundingDINO inference error: {e}")
            import traceback
            traceback.print_exc()
            self._publish_empty_detection(img_copy, header)
    
    def _init_tracking_with_sam2(self, img_bgr, boxes_np, labels, scores_np):
        """SAM2로 마스크 생성 및 트래킹 초기화"""
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            self.sam2_predictor.set_image(img_rgb)
            
            # SAM2 마스크 예측
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_np,
                multimask_output=False,
            )
            
            # 마스크 shape 정리 (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            elif masks.ndim == 3 and len(boxes_np) == 1:
                masks = masks[None]
            
            self.tracking_masks = masks
            self.tracking_boxes = boxes_np
            self.tracking_labels = list(labels)
            self.tracking_scores = scores_np.tolist() if hasattr(scores_np, 'tolist') else list(scores_np)
            
            self.get_logger().info(f"SAM2 tracking initialized with {len(masks)} masks")
            
        except Exception as e:
            self.get_logger().error(f"SAM2 mask generation error: {e}")
            self._init_tracking_bbox_only(boxes_np, labels, scores_np)
    
    def _init_tracking_bbox_only(self, boxes_np, labels, scores_np):
        """SAM2 없이 bbox만으로 트래킹 초기화"""
        self.tracking_masks = None
        self.tracking_boxes = boxes_np
        self.tracking_labels = list(labels)
        self.tracking_scores = scores_np.tolist() if hasattr(scores_np, 'tolist') else list(scores_np)
        self.get_logger().info(f"BBox-only tracking initialized with {len(boxes_np)} boxes")
    
    def _process_tracking(self, img_bgr: np.ndarray, header):
        """SAM2로 객체 트래킹"""
        if self.tracking_boxes is None or len(self.tracking_boxes) == 0:
            self.get_logger().warn("No tracking target, returning to IDLE")
            self._reset_state()
            self._publish_status("LOST")
            self.use_open_set = False
            return
        
        img_copy = img_bgr.copy()
        
        # SAM2가 있으면 마스크 트래킹
        if self.sam2_predictor is not None and self.tracking_masks is not None:
            try:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                self.sam2_predictor.set_image(img_rgb)
                
                # 이전 박스를 사용해서 새 마스크 예측
                masks, scores, logits = self.sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=self.tracking_boxes,
                    multimask_output=False,
                )
                
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                elif masks.ndim == 3 and len(self.tracking_boxes) == 1:
                    masks = masks[None]
                
                self.tracking_masks = masks
                
                # 마스크에서 새 bbox 추출
                new_boxes = []
                for i, mask in enumerate(masks):
                    # 마스크가 3D인 경우 2D로 변환
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    
                    ys, xs = np.where(mask)
                    if len(xs) > 0 and len(ys) > 0:
                        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
                        new_boxes.append([x1, y1, x2, y2])
                    elif i < len(self.tracking_boxes):
                        new_boxes.append(self.tracking_boxes[i])
                
                if new_boxes:
                    self.tracking_boxes = np.array(new_boxes)
                
            except Exception as e:
                self.get_logger().warn(f"SAM2 tracking error: {e}")
        
        # 시각화 및 퍼블리시
        self._visualize_and_publish(img_copy, header)
        self._publish_status("TRACKING")
    
    def _visualize_and_publish(self, img_copy: np.ndarray, header):
        """결과 시각화 및 토픽 퍼블리시 (세그멘테이션 + bbox 통합)"""
        detections_list = []
        H, W = img_copy.shape[:2]
        
        # 객체별 색상 팔레트 (BGR)
        color_palette = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 255, 0),  # Light green
            (255, 128, 0),  # Light blue
            (0, 128, 255),  # Orange
            (128, 0, 255),  # Purple
        ]
        
        # 세그멘테이션 마스크 저장 (포인트클라우드 생성용)
        self.current_segmentation_masks = []
        
        if self.tracking_boxes is not None:
            for i, box in enumerate(self.tracking_boxes):
                x1, y1, x2, y2 = [int(v) for v in box]
                label = self.tracking_labels[i] if i < len(self.tracking_labels) else "object"
                score = self.tracking_scores[i] if i < len(self.tracking_scores) else 0.0
                
                # 객체별 색상 선택
                color = color_palette[i % len(color_palette)]
                
                # Draw mask if available (세그멘테이션 오버레이)
                mask_2d = None
                if self.tracking_masks is not None and i < len(self.tracking_masks):
                    mask = self.tracking_masks[i]
                    
                    # 마스크가 3D인 경우 2D로 변환
                    if mask.ndim == 3:
                        mask = mask.squeeze()
                    
                    # 마스크 크기가 이미지와 다르면 리사이즈
                    if mask.shape[:2] != (H, W):
                        mask = cv2.resize(mask.astype(np.uint8), (W, H), 
                                         interpolation=cv2.INTER_NEAREST)
                    
                    mask_2d = mask.astype(bool)
                    
                    if mask_2d.any():
                        # 세그멘테이션 오버레이 (반투명 색상)
                        overlay_color = np.array(color, dtype=np.uint8)
                        img_copy[mask_2d] = (img_copy[mask_2d] * 0.5 + overlay_color * 0.5).astype(np.uint8)
                        
                        # 마스크 외곽선 그리기
                        contours, _ = cv2.findContours(mask_2d.astype(np.uint8), 
                                                       cv2.RETR_EXTERNAL, 
                                                       cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img_copy, contours, -1, color, 2)
                
                # 마스크 저장 (포인트클라우드용)
                self.current_segmentation_masks.append(mask_2d)
                
                # Draw bbox
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with background
                label_text = f"{label} {score:.2f}"
                (text_w, text_h), baseline = cv2.getTextSize(label_text, 
                                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_copy, (x1, max(0, y1-text_h-10)), 
                             (x1 + text_w, max(0, y1-5) + 5), color, -1)
                cv2.putText(img_copy, label_text, (x1, max(0, y1-5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Create Detection2D
                det = self._create_detection2d(x1, y1, x2, y2, label, score, header)
                detections_list.append(det)
        
        # Publish debug image (세그멘테이션 + bbox 통합)
        debug_msg = self.bridge.cv2_to_imgmsg(img_copy, 'bgr8')
        debug_msg.header = header
        self.debug_image_pub.publish(debug_msg)
        
        # Publish Detection2DArray
        det_array = Detection2DArray()
        det_array.header = header
        det_array.detections = detections_list
        self.detection_pub.publish(det_array)
        
        # Generate and publish PointCloud (if enabled)
        if self.enable_pointcloud:
            self._generate_and_publish_pointcloud(detections_list, header)
    
    def _generate_and_publish_pointcloud(self, detections: list, header):
        """세그멘테이션된 객체들의 3D 포인트클라우드 생성 및 퍼블리시"""
        # 필요한 데이터 체크
        if self.fx is None:
            self.get_logger().warn("Waiting for camera info...")
            return
        
        if self.latest_filtered_depth is None:
            self.get_logger().warn("Waiting for filtered depth image...")
            return
        
        if not hasattr(self, 'current_segmentation_masks') or not self.current_segmentation_masks:
            self.get_logger().debug("No segmentation masks available")
            return
        
        try:
            # Depth를 미터 단위로 변환
            depth_m = self.latest_filtered_depth.astype(np.float32) * self.depth_scale
            H, W = depth_m.shape[:2]
            
            all_points = []
            all_labels = []
            object_info_list = []
            
            for i, det in enumerate(detections):
                if not det.results:
                    continue
                
                class_id = det.results[0].hypothesis.class_id
                confidence = det.results[0].hypothesis.score
                object_id = i + 1
                
                # 해당 객체의 마스크 가져오기
                if i >= len(self.current_segmentation_masks):
                    continue
                
                object_mask = self.current_segmentation_masks[i]
                
                if object_mask is None or not object_mask.any():
                    # 마스크가 없으면 bbox 영역 사용
                    x1 = int(det.bbox.center.position.x - det.bbox.size_x / 2)
                    y1 = int(det.bbox.center.position.y - det.bbox.size_y / 2)
                    x2 = int(det.bbox.center.position.x + det.bbox.size_x / 2)
                    y2 = int(det.bbox.center.position.y + det.bbox.size_y / 2)
                    object_mask = np.zeros((H, W), dtype=bool)
                    object_mask[max(0,y1):min(H,y2), max(0,x1):min(W,x2)] = True
                
                # 마스크 크기 확인 및 조정
                if object_mask.shape[:2] != (H, W):
                    object_mask = cv2.resize(object_mask.astype(np.uint8), (W, H),
                                            interpolation=cv2.INTER_NEAREST).astype(bool)
                
                # 역투영으로 3D 포인트 생성
                points_3d = self._back_project(depth_m, object_mask)
                
                if len(points_3d) > 0:
                    all_points.append(points_3d)
                    all_labels.extend([object_id] * len(points_3d))
                    
                    # 객체 중심점 계산
                    centroid = np.mean(points_3d, axis=0)
                    
                    object_info_list.append({
                        'id': object_id,
                        'class': class_id,
                        'confidence': float(confidence),
                        'num_points': len(points_3d),
                        'centroid': {
                            'x': float(centroid[0]),
                            'y': float(centroid[1]),
                            'z': float(centroid[2])
                        }
                    })
            
            # 포인트클라우드 퍼블리시
            if all_points:
                combined_points = np.vstack(all_points)
                combined_labels = np.array(all_labels, dtype=np.int32)
                
                # PointCloud2 메시지 생성
                pc_header = Header()
                pc_header.stamp = self.latest_depth_header.stamp if self.latest_depth_header else header.stamp
                pc_header.frame_id = self.camera_frame if self.camera_frame else "camera_link"
                
                pc_msg = self._create_pointcloud2(combined_points, combined_labels, pc_header)
                self.pointcloud_pub.publish(pc_msg)
                
                # 객체 정보 퍼블리시
                info_str = json.dumps(object_info_list, indent=2)
                self.object_info_pub.publish(String(data=info_str))
                
                self.get_logger().debug(
                    f"Published PointCloud: {len(combined_points)} points, {len(object_info_list)} objects")
                    
        except Exception as e:
            self.get_logger().error(f"PointCloud generation error: {e}")
            import traceback
            traceback.print_exc()
    
    def _back_project(self, depth_m: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        역투영(Back-projection): 2D 픽셀 좌표 + depth -> 3D 카메라 좌표
        
        수식:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = depth
        """
        # 마스크 영역의 픽셀 좌표
        v, u = np.where(mask)
        z = depth_m[v, u]
        
        # 유효한 depth 필터링
        valid = (z > self.min_depth) & (z < self.max_depth)
        u = u[valid]
        v = v[valid]
        z = z[valid]
        
        if len(z) == 0:
            return np.array([]).reshape(0, 3)
        
        # 3D 좌표 계산
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        points = np.stack([x, y, z], axis=1)
        return points
    
    def _create_pointcloud2(self, points: np.ndarray, labels: np.ndarray, 
                            header: Header) -> PointCloud2:
        """
        PointCloud2 메시지 생성 (XYZ + Label)
        
        Fields:
        - x, y, z: 3D 좌표 (float32)
        - label: 객체 ID (int32)
        """
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='label', offset=12, datatype=PointField.INT32, count=1),
        ]
        
        point_step = 16  # 4 fields * 4 bytes
        row_step = point_step * len(points)
        
        # 데이터 패킹
        data = []
        for i in range(len(points)):
            x, y, z = points[i]
            label = labels[i]
            data.append(struct.pack('fffi', x, y, z, label))
        
        cloud_data = b''.join(data)
        
        pc_msg = PointCloud2(
            header=header,
            height=1,
            width=len(points),
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=point_step,
            row_step=row_step,
            data=cloud_data
        )
        
        return pc_msg
    
    def _publish_empty_detection(self, img_copy: np.ndarray, header):
        """빈 detection 퍼블리시"""
        debug_msg = self.bridge.cv2_to_imgmsg(img_copy, 'bgr8')
        debug_msg.header = header
        self.debug_image_pub.publish(debug_msg)
        
        det_array = Detection2DArray()
        det_array.header = header
        det_array.detections = []
        self.detection_pub.publish(det_array)
    
    def _create_detection2d(self, x1, y1, x2, y2, label, confidence, header):
        """Detection2D 메시지 생성"""
        det = Detection2D()
        det.header = header
        
        det.bbox.center.position.x = float((x1 + x2) / 2)
        det.bbox.center.position.y = float((y1 + y2) / 2)
        det.bbox.size_x = float(x2 - x1)
        det.bbox.size_y = float(y2 - y1)
        
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = label
        hyp.hypothesis.score = float(confidence)
        det.results.append(hyp)
        
        return det


def main(args=None):
    rclpy.init(args=args)
    node = GroundedSAM2Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
