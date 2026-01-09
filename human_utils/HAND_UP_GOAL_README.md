# Hand Up Goal System

HANDS_UP 상태의 사람을 감지하여 그 사람 방향으로 nav2 목적지를 생성하는 시스템입니다. **두 가지 버전**이 제공됩니다.

## 버전 비교

### 1. Monitor 버전 (모니터링 전용)
- **파일**: `hand_up_goal_monitor.py`
- **노드**: `hand_up_goal_monitor_node`
- **특징**:
  - Goal을 **한 번만** 발행
  - 거리 모니터링하여 **임계값까지 얼마나 남았는지 프린트만**
  - 자동 취소 없음 (수동 제어)
  - 사용자가 직접 로봇을 멈추거나 제어
- **용도**: 테스트, 수동 제어 환경

### 2. Controller 버전 (자동 제어)
- **파일**: `hand_up_goal_controller.py`
- **노드**: `hand_up_goal_controller_node`
- **특징**:
  - Goal 발행 후 지속적으로 거리 모니터링
  - 임계값 이하로 접근하면 **자동으로 navigation 취소**
  - 사람이 사라지면 자동 정지
  - 완전 자율 주행
- **용도**: 실제 운영 환경, 자율 주행

## 기능

1. **HANDS_UP 상태 감지**: `human_state_detector_node`에서 발행하는 `/human/states` 토픽을 구독하여 HAND_UP_LEFT, HAND_UP_RIGHT, HAND_UP_BOTH 상태 감지
2. **가장 가까운 스켈레톤 점 찾기**: 17개의 스켈레톤 점 중 원점(로봇)에서 가장 가까운 점 선택
3. **목적지 계산**: 가장 가까운 점에서 1.2m 앞(로봇 방향) 좌표 계산
4. **좌표 변환**: TF를 사용하여 skeleton_frame → map_frame으로 변환
5. **Nav2 Action 호출**: `NavigateToPose` action으로 목적지 전송
6. **거리 모니터링**: 실시간으로 스켈레톤 거리 체크, 임계값(0.8m) 이하면 자동 정지
7. **시각화**: `/hand_up_goal_marker` 토픽으로 rviz 마커 발행

## 노드 정보

### 공통 기능
1. **HANDS_UP 상태 감지**: HAND_UP_LEFT, HAND_UP_RIGHT, HAND_UP_BOTH
2. **가장 가까운 스켈레톤 점 찾기**: 17개 점 중 원점에서 가장 가까운 점
3. **목적지 계산**: 가장 가까운 점에서 1.2m 앞 좌표
4. **좌표 변환**: skeleton_frame → map_frame
5. **Nav2 Action 호출**: NavigateToPose
6. **RViz 시각화**: 마커 표시
7. **🔄 리셋 기능**: `/hand_up_goal/reset` 토픽으로 시스템 리셋
8. **🛡️ 강건성 향상**:
   - **연속 N프레임 이상 손 든 사람이 감지되어야 발행** (기본 5프레임)
   - 잠깐 잘못 감지되는 것 방지 (False Positive 제거)
   - 사람이 아예 없을 때 goal 찍히는 문제 해결

### hand_up_goal_monitor_node (모니터링 버전)

#### 동작 방식
- Goal을 **한 번만** 발행
- 2Hz 주기로 거리 프린트:
  ```
  📍 Person 0: Distance=2.45m | Remaining to threshold: 1.65m
  ⚠️  Person 0: Distance=0.95m | Remaining: 0.15m | ALMOST THERE!
  🛑 Person 0: Distance=0.70m | REACHED THRESHOLD | Over by 0.10m
  ```
- **자동 취소 없음** - 사용자가 수동으로 제어

#### 구독 토픽
- `/human/states` (std_msgs/String)
- `/human/skeleton_markers` (visualization_msgs/MarkerArray)

#### 발행 토픽
- `/hand_up_goal` (geometry_msgs/PoseStamped)
- `/hand_up_goal_marker` (visualization_msgs/Marker) - 초록색 화살표

#### 구독 토픽 (리셋용)
- `/hand_up_goal/reset` (std_msgs/String) - 아무 문자열이나 발행하면 리셋

#### 파라미터
- `map_frame` (string, default: "map")
- `goal_distance` (double, default: 1.2)
- `min_skeleton_points` (int, default: 5)
- `stop_distance` (double, default: 0.8) - 경고 표시 임계값
- `distance_check_rate` (double, default: 2.0) - 프린트 주기 (Hz)
- `use_nav2` (bool, default: true)
- **`min_detection_frames`** (int, default: 5) - **최소 연속 감지 프레임 수** (False Positive 방지)

### hand_up_goal_controller_node (자동 제어 버전)

### hand_up_goal_controller_node (자동 제어 버전)

#### 동작 방식
- Goal 발행 후 지속적으로 모니터링
- 5Hz 주기로 거리 체크
- **거리 ≤ stop_distance이면 자동으로 navigation 취소**
- 사람 사라지면 자동 취소

#### 구독 토픽
- `/human/states` (std_msgs/String)
- `/human/skeleton_markers` (visualization_msgs/MarkerArray)

#### 발행 토픽  
- `/hand_up_goal` (geometry_msgs/PoseStamped)
- `/hand_up_goal_marker` (visualization_msgs/Marker) - 빨간색 화살표

#### 발행 토픽  
- `/hand_up_goal` (geometry_msgs/PoseStamped)
- `/hand_up_goal_marker` (visualization_msgs/Marker) - 빨간색 화살표

#### 구독 토픽 (리셋용)
- `/hand_up_goal/reset` (std_msgs/String) - 아무 문자열이나 발행하면 리셋

#### 파라미터
- `map_frame` (string, default: "map")
- `goal_distance` (double, default: 1.2)
- `min_skeleton_points` (int, default: 5)
- `stop_distance` (double, default: 0.8) - **자동 정지 임계값**
- `distance_check_rate` (double, default: 5.0) - 체크 주기 (Hz)
- `use_nav2` (bool, default: true)
- **`min_detection_frames`** (int, default: 5) - **최소 연속 감지 프레임 수** (False Positive 방지)

## 빌드 및 실행

### 빌드
```bash
cd /home/nvidia/vision_ws
colcon build --packages-select human_utils
source install/setup.bash
```

### 개별 노드 실행

```bash
# Terminal 1: Human State Detector
ros2 run human_utils human_state_detector_node

# Terminal 2-A: Monitor 버전 (거리 모니터링만)
ros2 run human_utils hand_up_goal_monitor_node

# 또는

# Terminal 2-B: Controller 버전 (자동 제어)
ros2 run human_utils hand_up_goal_controller_node
```

### Launch 파일로 실행 (권장)

```bash
# Monitor 버전 (거리 프린트만, 자동 취소 없음)
ros2 launch human_utils hand_up_goal.launch.py

# Controller 버전 (자동 정지)
ros2 launch human_utils hand_up_goal_auto.launch.py
```

## 사용 시나리오

### 시나리오 1: 테스트 / 수동 제어 (Monitor 버전)
```bash
ros2 launch human_utils hand_up_goal.launch.py
```

**동작**:
1. 손 든 사람을 **5프레임 이상 연속** 감지 → Goal 한 번 발행
   ```
   ⏳ Detecting person 0: 1/5 frames
   ⏳ Detecting person 0: 3/5 frames
   ✅ Person 0 consistently detected! Sending goal.
   ```
2. 터미널에 거리 정보 프린트:
   ```
   📍 Person 0: Distance=3.20m | Remaining to threshold: 2.40m
   ⚠️  Person 0: Distance=0.95m | ALMOST THERE!
   🛑 Person 0: Distance=0.75m | REACHED THRESHOLD
   ```
3. 로봇은 계속 주행 (자동 정지 안 함)
4. **사용자가 수동으로 멈추거나 제어**

**리셋**:
```bash
# 새로운 사람 감지를 위해 시스템 리셋
ros2 topic pub --once /hand_up_goal/reset std_msgs/String "data: 'reset'"
```

### 시나리오 2: 자율 주행 (Controller 버전)
```bash
ros2 launch human_utils hand_up_goal_auto.launch.py
```

**동작**:
1. 손 든 사람을 **5프레임 이상 연속** 감지 → Goal 발행
2. 로봇 주행 시작
3. 5Hz로 거리 체크
4. **거리 ≤ 0.8m → 자동으로 navigation 취소 및 정지**
5. 안전하게 사람 앞에 도착

**리셋**:
```bash
# Navigation 취소 및 시스템 리셋
ros2 topic pub --once /hand_up_goal/reset std_msgs/String "data: 'reset'"
```

## 강건성 향상 기능

### 1. 연속 프레임 감지 (Temporal Filtering) ⭐
**핵심 기능: 사람이 없을 때 잘못 감지되는 문제 해결**

- 손을 든 사람이 **최소 5프레임 이상 연속으로 감지**되어야 goal 발행
- 잠깐 잘못 감지되는 것(False Positive) 방지
- 실시간 카운터: `⏳ Detecting person 0: 3/5 frames`

**왜 필요한가?**
- YOLO 모델이 가끔 사람이 없는데도 잘못 감지
- 카메라 노이즈나 배경 물체를 사람으로 오인
- 연속 5프레임 감지로 확실한 경우만 goal 발행

### 2. 리셋 기능
```bash
# Monitor 버전: goal_sent 플래그 리셋
ros2 topic pub --once /hand_up_goal/reset std_msgs/String "data: 'reset'"

# Controller 버전: 현재 navigation 취소 + 시스템 리셋
ros2 topic pub --once /hand_up_goal/reset std_msgs/String "data: 'reset'"
```

### 3. 중복 방지
- **Monitor 버전**: goal을 한 번 보내면 리셋 전까지 다시 안 보냄
- **Controller 버전**: 같은 사람에게 중복으로 goal 안 보냄

## RViz 시각화

RViz에서 다음 토픽들을 추가하세요:

1. **MarkerArray** - `/human/skeleton_markers`: 사람 스켈레톤
2. **Image** - `/human/debug_image`: 디버그 이미지
3. **Marker** - `/hand_up_goal_marker`: 
   - **초록색 화살표**: Monitor 버전
   - **빨간색 화살표**: Controller 버전
4. **PoseStamped** - `/hand_up_goal`: 목적지 포즈

## 동작 원리

### 1. 목적지 계산
1. HANDS_UP 상태인 사람들 중 가장 가까운 사람 선택
2. 그 사람의 17개 스켈레톤 점 중 원점에서 가장 가까운 점 찾기
3. 그 점에서 원점(로봇) 방향으로 `goal_distance`(기본 1.2m) 앞 좌표 계산
4. TF를 통해 map 좌표계로 변환

### 2. Navigation 실행
1. `NavigateToPose` action으로 Nav2에 목적지 전송
2. Nav2가 경로 계획 및 주행 시작

### 3-A. 거리 모니터링 (Monitor 버전)
1. `distance_check_rate`(기본 2Hz) 주기로 거리 체크
2. 터미널에 거리 정보 프린트만
3. **자동 취소 없음**

### 3-B. 거리 모니터링 & 자동 정지 (Controller 버전)
1. `distance_check_rate`(기본 5Hz) 주기로 타겟 사람과의 거리 체크
2. 스켈레톤의 최소 거리가 `stop_distance`(기본 0.8m) 이하가 되면:
   - **Navigation goal 자동 취소**
   - 로봇 정지
3. 타겟 사람이 시야에서 사라지면:
   - Navigation goal 자동 취소

### 안전 기능 (Controller 버전)
- ✅ 실시간 거리 모니터링으로 충돌 방지
- ✅ 사람 사라짐 감지하여 자동 정지
- ✅ 임계값 이하 접근 시 자동 정지
- ✅ Nav2 action feedback 수신

## 좌표계 설명

### 스켈레톤 프레임 (마커의 frame_id 자동 감지)
- 마커가 어떤 프레임을 사용하든 자동으로 처리
- 일반적으로 `camera_color_optical_frame` 또는 `base` 프레임

### 목적지 계산 방식
1. 스켈레톤의 17개 점 중 원점(0,0,0)에서 가장 가까운 점 선택
2. 그 점에서 원점 방향으로 1.2m 앞 좌표 계산
   - `goal = person_point - 1.2 * direction_unit_vector`
3. TF를 통해 map 좌표계로 변환
4. 지면 레벨(z=0)로 조정하여 발행

## 의존성

- ROS2 Humble
- nav2_msgs
- tf2_ros
- tf2_geometry_msgs
- geometry_msgs
- visualization_msgs
- std_msgs

## 참고사항

- 여러 명이 손을 들고 있으면 가장 가까운 사람 선택
- TF 변환이 실패하면 목적지 발행 안 됨
- 스켈레톤 점이 5개 미만이면 무시됨
- **Monitor 버전**: Goal 한 번만 발행, 자동 취소 없음
- **Controller 버전**: 거리 모니터링하여 자동 정지
- `use_nav2=false`로 설정하면 PoseStamped만 발행 (테스트용)

## 버전 선택 가이드

| 상황 | 추천 버전 |
|------|----------|
| 테스트 중 | Monitor |
| 수동으로 멈추고 싶을 때 | Monitor |
| 완전 자율 주행 | Controller |
| 안전이 중요한 실제 환경 | Controller |
| 로봇이 더 가까이 가도 괜찮을 때 | Monitor |
