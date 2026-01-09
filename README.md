# inha_perception

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
  '''
  data: P0:STANDING, P1:UNKNOWN
  '''
- /human/skeleton_markers
  '''
  ./topic_example.txt
  '''


To Be modifed
- 
- change depth information from camera's depth to 3D LiDAR
- Waving state
- robust if else

