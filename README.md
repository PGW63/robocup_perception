# inha_perception

## perception - Detection
Node
- YOLO node
- Grounded SAM2 node
- Depth EMA node

### YOLO node
Topic
subscription
- /camera/camera/color/image_raw/compressed (Image) 
- /detection_node/use_open_set (Bool) : True일시 Grounded SAM2 노드가 활성화되고 False이면 YOLO 노드가 활성화

publisher
- /detection_node/debug_image (Image) : 디버깅용 이미지 
- /detection_node/detections (Detection2DArray of vision_msgs) : 현재 이미지에서 디텍션된 물체들을 배열로 넘겨줌

### Grounded SAM2 node
Topic 
subscription
- /camera/camera/color/image_raw/compressed (Image)
- /camera/camera/aligned_depth_to_color/camera_info (CameraInfo)
- /detection_node/filtered_depth_image (Image) : 카메라 뎁스를 필터링해서 안정화 시킨 이미지
- /detection_node/use_open_set (Bool) : GroundedSAM2 node를 사용할지 YOLO node를 사용할지
- /detection_node/search (String) : openSet Detection을 위해 찾을 물체들 (입력예시: coke. banana. green apple. ) - GroundingDino에서 제안하는 프롬프팅
- /detection_node/stop (Bool) : 한번 찾으면 SAM2로 트래킹을 쭉 하는데, 트래킹을 끊기 위한 트리거

publisher
- /detection_node/status (String) : 현재 디텍션 상태 (IDLE, SEARCHING, TRACKING)
- /detection_node/use_open_set (Bool) : 이전과 동일
- /detection_node/debug_image (Image) : 디버깅용 이미지
- /detection_node/detections (Detection2DArray of vision_msgs) : 이전과 동일
- /detection_node/segmented_pointcloud (PointCloud2) : 찾은 물체에 대해서 포인트클라우드로 발행
- /detection_node/object_info (String) : detection 토픽과는 다르게 3차원 정보에 대한 토픽

### Depth EMA node
Topic
subscription
- /camera/camera/aligned_depth_to_color/image_raw (Image)

publisher
- /detection_node/filtered_depth (Image) : 필터를 거친 뎁스이미지

### /detection_node/detections 예시
```
header:
  stamp:
    sec: 1710000000
    nanosec: 123456789
  frame_id: "camera_color_frame"
detections:
- bbox:
    center:
      position: {x: 320.0, y: 240.0}
      theta: 0.0
    size_x: 120.0
    size_y: 80.0
  results:
  - hypothesis:
      class_id: "apple"
      score: 0.92
```

### /detection_node/object_info 예시
```
[
  {
    "id": 1,
    "class": "apple",
    "confidence": 0.92,
    "num_points": 1543,
    "centroid": {
      "x": 0.42,
      "y": -0.08,
      "z": 0.73
    }
  }
]
```

## human utils - Pose Detection

### humamn state detection
State
- UNKNOWN
- STANDING
- SITTING
- LYING_DOWN
- HAND_UP_LEFT
- HAND_UP_RIGHT
- HAND_UP_BOTH

TOPIC
subscription
- /camera/camera/color/image_raw (Image)
- /camera/camera/color/camera_info (CamInfo)
- /camera/camera/aligned_depth_to_color/image_raw (Image) 
- /human/resume (Bool)

publisher
- /human/states (String) : 위의 State 중 하나에 해당되는 것을 발행
- /human/skeleton_markers (MarkerArray) : 스켈레톤의 좌표 및 State 발행
- /human/debug_image (Image) : 시각화 디버깅 이미지

topic example
- /human/states
  ```
  data: P0:STANDING, P1:UNKNOWN
  ```
- /human/skeleton_markers
  ```
  ./topic_example.txt
  ```


To Be modifed
- 
- change depth information from camera's depth to 3D LiDAR
- Waving state
- robust if else

