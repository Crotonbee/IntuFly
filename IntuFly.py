import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import math
from l2cs import Pipeline
from collections import deque
import time
import json
import threading
from threading import Lock, Event
import pandas as pd
import asyncio
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw, PositionNedYaw, VelocityBodyYawspeed
import mediapipe as mp
import matplotlib.cm as cm
from filterpy.kalman import KalmanFilter
import queue
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

MODEL_FILE = "model.pt"
SMOOTH_K = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, in_dim=63, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        return self.net(x)


class GestureModelWrapper:
    def __init__(self, model_path):
        if not pathlib.Path(model_path).exists():
            raise FileNotFoundError(f"手势识别模型文件 {model_path} 未找到")

        print(f"正在加载手势识别模型: {model_path}")
        ck = torch.load(model_path, map_location=DEVICE)
        labels = ck.get("labels", None)
        state = ck.get("model_state", None)
        if state is None:
            state = ck

        if labels is None:
            labels = {"1": 0, "ok": 1, "thumb": 2, "noges": 3, "fist": 4, "brake": 5}
            n_classes = 6
            print("使用默认手势标签映射")
        else:
            n_classes = len(labels)
            print(f"从模型加载标签映射: {labels}")

        model = MLP(in_dim=63, n_classes=n_classes)
        try:
            model.load_state_dict(state)
            print("模型权重加载成功")
        except RuntimeError as e:
            print(f"模型权重加载失败，尝试键名映射: {e}")
            sd = {}
            has_net_prefix = any(k.startswith("net.") for k in state.keys())
            expect_net_prefix = any(k.startswith("net.") for k in model.state_dict().keys())

            if has_net_prefix and not expect_net_prefix:
                print("移除 'net.' 前缀")
                for k, v in state.items():
                    if k.startswith("net."):
                        sd[k[len("net."):]] = v
                    else:
                        sd[k] = v
            elif (not has_net_prefix) and expect_net_prefix:
                print("添加 'net.' 前缀")
                for k, v in state.items():
                    sd["net." + k] = v
            else:
                sd = state

            model.load_state_dict(sd)

        model.to(DEVICE)
        model.eval()
        self.model = model
        self.labels = labels
        self.id2label = {int(v): k for k, v in labels.items()}

    def predict(self, x_np):
        x = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = self.model(x)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
        return probs

    def extract_landmarks(self, hand_landmarks):
        if hasattr(hand_landmarks, 'landmark'):
            landmarks = hand_landmarks.landmark
        else:
            landmarks = hand_landmarks

        try:
            pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float32)
            rel = pts - pts[0:1, :]
            maxd = np.max(np.linalg.norm(rel, axis=1)) + 1e-8
            return (rel / maxd).reshape(-1)
        except Exception as e:
            print(f"特征提取错误: {e}")
            print(f"Landmarks类型: {type(landmarks)}")
            if hasattr(landmarks, '__len__'):
                print(f"Landmarks长度: {len(landmarks)}")
            raise

@dataclass
class GazeData:
    timestamp: float
    gaze_point: Optional[Tuple[float, float]]
    img_gaze_point: Optional[Tuple[float, float]]
    gazed_object: Optional[Dict[str, Any]]
    eye_pos_3d: Optional[list]
    motion_state: str
    smoothed_pitch: float
    smoothed_yaw: float
    raw_pitch: float
    raw_yaw: float
    corrected_pitch: float
    corrected_yaw: float
    face_bbox: Optional[Tuple[float, float, float, float]]
    face_score: float
    tracking_status: str
    yaw_control_mode: str
    current_yaw_speed: float
    calibration_mode: bool
    realsense_connected: bool

async def init_drone():
    try:
        print("开始无人机初始化（等待ok手势起飞)")
        drone = System()
        await drone.connect(system_address="udp://:14540")

        async for state in drone.core.connection_state():
            if state.is_connected:
                print("✓ 无人机已连接")
                break

        await drone.action.arm()
        print("✓ 无人机已解锁")

        await drone.offboard.set_position_ned(
            PositionNedYaw(north_m=0.0, east_m=0.0, down_m=0.0, yaw_deg=0.0)
        )

        await drone.offboard.start()
        print("✓ Offboard模式已启动，等待ok手势起飞")
        return drone, True

    except Exception as e:
        print(f"✗ 无人机初始化失败: {e}")
        return None, False

async def takeoff_drone(drone, target_altitude=-3.0):
    try:
        print(f"检测到ok手势，开始起飞到{abs(target_altitude)}米高度...")
        start_time = time.time()
        ascent_duration = 8.0

        while (time.time() - start_time) < ascent_duration:
            progress = (time.time() - start_time) / ascent_duration
            current_altitude = target_altitude * progress
            await drone.offboard.set_position_ned(
                PositionNedYaw(north_m=0.0, east_m=0.0, down_m=current_altitude, yaw_deg=90.0)
            )
            await asyncio.sleep(0.1)

        print("✓ 无人机起飞完成")
        return True
    except Exception as e:
        print(f"✗ 无人机起飞失败: {e}")
        return False

class InstantGazeObjectMatcher:
    def __init__(self, history_size=10, dwell_threshold=5, margin=20):
        self.history = deque(maxlen=history_size)
        self.dwell_threshold = dwell_threshold
        self.margin = margin
        self.stable_object = None
        self.fast_mode = True

    def is_point_in_bbox(self, point, bbox):
        if not point:
            return False
        x, y = point
        x1, y1, x2, y2 = bbox
        return (x1 - self.margin <= x <= x2 + self.margin and
                y1 - self.margin <= y <= y2 + self.margin)

    def find_gazed_object(self, gaze_point, detected_objects):
        if not gaze_point or not detected_objects:
            self.history.append(None)
            return None

        gazed_objects = [obj for obj in detected_objects
                         if self.is_point_in_bbox(gaze_point, obj['bbox'])]
        current_gazed = max(gazed_objects, key=lambda x: x['confidence']) if gazed_objects else None
        self.history.append(current_gazed)

        if self.fast_mode and current_gazed:
            return current_gazed

        if len(self.history) >= self.dwell_threshold:
            recent_objects = list(self.history)[-self.dwell_threshold:]
            if all(obj is not None for obj in recent_objects):
                class_names = [obj['class_name'] for obj in recent_objects]
                if len(set(class_names)) == 1:
                    self.stable_object = current_gazed
                    return current_gazed

        return self.stable_object

    def set_fast_mode(self, enabled):
        self.fast_mode = enabled
        print(f"即时响应模式: {'开启' if enabled else '关闭'}")

    def reset(self):
        self.history.clear()
        self.stable_object = None

class ObjectTracker:
    def __init__(self, lock_threshold_frames=60, lock_threshold_seconds=2.0):
        self.lock_threshold_frames = lock_threshold_frames
        self.lock_threshold_seconds = lock_threshold_seconds
        self.current_target = None
        self.target_frames_count = 0
        self.target_start_time = None
        self.is_locked = False
        self.center_deadzone = 40
        self.tracking_yaw_scale = 0.03
        self.max_tracking_yaw_speed = 30
        self.last_gazed_time = None
        self.gaze_timeout_seconds = 2
    def update(self, gazed_object, detected_objects, frame_width):
        current_time = time.time()

        if gazed_object:
            if self.current_target is None:
                self._set_new_target(gazed_object, current_time)
            elif self._is_same_object(self.current_target, gazed_object):
                self._update_existing_target(gazed_object, current_time)
                if self.is_locked:
                    self.last_gazed_time = current_time
            else:
                self._handle_target_change(gazed_object, current_time)
        else:
            self._handle_no_gaze(detected_objects)

    def _set_new_target(self, gazed_object, current_time):
        self.current_target = gazed_object
        self.target_frames_count = 1
        self.target_start_time = current_time
        self.is_locked = False
        self.last_gazed_time = current_time

    def _update_existing_target(self, gazed_object, current_time):
        self.target_frames_count += 1
        time_condition = (current_time - self.target_start_time) >= self.lock_threshold_seconds
        frame_condition = self.target_frames_count >= self.lock_threshold_frames

        if (time_condition or frame_condition) and not self.is_locked:
            self.is_locked = True
            print(f"物体锁定：目标: {gazed_object['class_name']}")

        self.current_target = gazed_object

    def _handle_target_change(self, gazed_object, current_time):
        if self.is_locked:
            self._handle_locked_target_change(gazed_object, current_time)
        else:
            self._set_new_target(gazed_object, current_time)

    def _handle_locked_target_change(self, gazed_object, current_time):
        if not hasattr(self, 'new_target_candidate') or not self._is_same_object(self.new_target_candidate,
                                                                                 gazed_object):
            self.new_target_candidate = gazed_object
            self.new_target_frames = 1
            self.new_target_start_time = current_time
        else:
            self.new_target_frames += 1
            if self._should_switch_target(current_time):
                self._switch_to_new_target(gazed_object, current_time)

    def _should_switch_target(self, current_time):
        time_condition = (current_time - self.new_target_start_time) >= self.lock_threshold_seconds
        frame_condition = self.new_target_frames >= self.lock_threshold_frames
        return time_condition or frame_condition

    def _switch_to_new_target(self, gazed_object, current_time):
        self.current_target = gazed_object
        self.target_frames_count = self.new_target_frames
        self.target_start_time = self.new_target_start_time
        self.is_locked = True
        self._clear_candidate()

    def _clear_candidate(self):
        for attr in ['new_target_candidate', 'new_target_frames', 'new_target_start_time']:
            if hasattr(self, attr):
                delattr(self, attr)

    def _handle_no_gaze(self, detected_objects):
        current_time = time.time()
        self._clear_candidate()
        if self.is_locked and self.current_target:

            if (self.last_gazed_time is not None and
                    current_time - self.last_gazed_time > self.gaze_timeout_seconds):
                print(f"超过{self.gaze_timeout_seconds}秒未注视锁定目标，自动解锁: {self.current_target['class_name']}")
                self.reset()
                return

            if not self._target_still_visible(detected_objects):
                print(f"锁定目标消失: {self.current_target['class_name']}")
                self.reset()
        elif self.current_target is not None:
            self.reset()

    def _target_still_visible(self, detected_objects):
        for obj in detected_objects:
            if self._is_same_object(self.current_target, obj):
                self.current_target = obj
                return True
        return False

    def calculate_tracking_yaw_speed(self, frame_width):
        if not self.is_locked or not self.current_target:
            return 0.0

        bbox = self.current_target['bbox']
        object_center_x = (bbox[0] + bbox[2]) / 2
        frame_center_x = frame_width / 2
        x_offset = object_center_x - frame_center_x

        if abs(x_offset) <= self.center_deadzone:
            return 0.0

        effective_offset = x_offset - self.center_deadzone if x_offset > 0 else x_offset + self.center_deadzone
        yaw_speed = effective_offset * self.tracking_yaw_scale
        return np.clip(yaw_speed, -self.max_tracking_yaw_speed, self.max_tracking_yaw_speed)

    def _is_same_object(self, obj1, obj2):
        if not obj1 or not obj2 or obj1['class_name'] != obj2['class_name']:
            return False

        center1 = ((obj1['bbox'][0] + obj1['bbox'][2]) / 2, (obj1['bbox'][1] + obj1['bbox'][3]) / 2)
        center2 = ((obj2['bbox'][0] + obj2['bbox'][2]) / 2, (obj2['bbox'][1] + obj2['bbox'][3]) / 2)
        distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance < 100

    def reset(self):
        self.current_target = None
        self.target_frames_count = 0
        self.target_start_time = None
        self.is_locked = False
        self._clear_candidate()
        self.last_gazed_time = None

    def get_status(self):
        if not self.current_target:
            return "无目标"

        if self.is_locked:
            status = f"锁定: {self.current_target['class_name']}"
            if hasattr(self, 'new_target_candidate'):
                progress = self._get_candidate_progress()
                status += f"候选: {self.new_target_candidate['class_name']} ({progress * 100:.0f}%)"
            return status
        else:
            progress = self._get_current_progress()
            return f"跟踪: {self.current_target['class_name']} ({progress * 100:.0f}%)"

    def _get_candidate_progress(self):
        if not hasattr(self, 'new_target_start_time'):
            return 0.0
        elapsed = time.time() - self.new_target_start_time
        time_progress = min(elapsed / self.lock_threshold_seconds, 1.0)
        frame_progress = min(getattr(self, 'new_target_frames', 0) / self.lock_threshold_frames, 1.0)
        return max(time_progress, frame_progress)

    def _get_current_progress(self):
        frame_progress = min(self.target_frames_count / self.lock_threshold_frames, 1.0)
        time_progress = 0.0
        if self.target_start_time:
            elapsed = time.time() - self.target_start_time
            time_progress = min(elapsed / self.lock_threshold_seconds, 1.0)
        return max(frame_progress, time_progress)

class CameraCalibrator:
    def __init__(self, screen_width, screen_height):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.calibration_points = []
        self.measured_angles = []
        self.camera_pitch_offset = 0.0
        self.camera_yaw_offset = 0.0
        self.is_calibrated = False
        self.calibration_file = "camera_calibration.json"

    def get_calibration_points(self):
        margin = 0.1
        points = []
        for y_ratio in [margin, 0.5, 1 - margin]:
            for x_ratio in [margin, 0.5, 1 - margin]:
                points.append((int(x_ratio * self.screen_width), int(y_ratio * self.screen_height)))
        return points

    def add_calibration_data(self, screen_point, measured_pitch, measured_yaw, eye_pos_3d):
        self.calibration_points.append(screen_point)
        self.measured_angles.append((measured_pitch, measured_yaw, eye_pos_3d))

    def calculate_camera_pose(self):
        if len(self.calibration_points) < 5:
            return False

        pitch_errors, yaw_errors = [], []
        for i, (screen_x, screen_y) in enumerate(self.calibration_points):
            measured_pitch, measured_yaw, eye_pos_3d = self.measured_angles[i]
            if eye_pos_3d:
                theoretical_angles = self._calculate_theoretical_angles(screen_x, screen_y, eye_pos_3d)
                if theoretical_angles[0] is not None:
                    pitch_errors.append(measured_pitch - theoretical_angles[0])
                    yaw_errors.append(measured_yaw - theoretical_angles[1])

        if len(pitch_errors) < 3:
            return False

        self.camera_pitch_offset = np.mean(pitch_errors)
        self.camera_yaw_offset = np.mean(yaw_errors)
        self.is_calibrated = True
        self.save_calibration()
        print(f"摄像机校准完成: 俯仰偏移={self.camera_pitch_offset:.2f}°, 航向偏移={self.camera_yaw_offset:.2f}°")
        return True

    def _calculate_theoretical_angles(self, screen_x, screen_y, eye_pos_3d):
        try:
            eye_x, eye_y, eye_z = eye_pos_3d
            screen_width_mm, screen_height_mm = 596.74, 335.66
            camera_offset_y = -167.83 + 70

            screen_point_x = (screen_x / self.screen_width - 0.5) * screen_width_mm
            screen_point_y = (0.5 - screen_y / self.screen_height) * screen_height_mm

            dx = screen_point_x - eye_x
            dy = screen_point_y - (-eye_y + camera_offset_y)
            dz = -eye_z

            distance_xz = math.sqrt(dx * dx + dz * dz)
            theoretical_pitch = math.degrees(math.atan2(dy, distance_xz))
            theoretical_yaw = math.degrees(math.atan2(dx, -dz))

            return theoretical_pitch, theoretical_yaw
        except:
            return None, None

    def correct_angles(self, raw_pitch, raw_yaw):
        if not self.is_calibrated:
            return raw_pitch, raw_yaw
        return raw_pitch - self.camera_pitch_offset, raw_yaw - self.camera_yaw_offset

    def save_calibration(self):
        data = {
            'camera_pitch_offset': self.camera_pitch_offset,
            'camera_yaw_offset': self.camera_yaw_offset,
            'is_calibrated': self.is_calibrated
        }
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            print(f"保存校准数据失败: {e}")

    def load_calibration(self):
        try:
            with open(self.calibration_file, 'r') as f:
                data = json.load(f)
            self.camera_pitch_offset = data.get('camera_pitch_offset', 0.0)
            self.camera_yaw_offset = data.get('camera_yaw_offset', 0.0)
            self.is_calibrated = data.get('is_calibrated', False)
            if self.is_calibrated:
                print(
                    f"已加载校准数据: 俯仰偏移={self.camera_pitch_offset:.2f}°, 航向偏移={self.camera_yaw_offset:.2f}°")
            return True
        except:
            return False

    def reset_calibration(self):
        self.calibration_points = []
        self.measured_angles = []
        self.camera_pitch_offset = 0.0
        self.camera_yaw_offset = 0.0
        self.is_calibrated = False


class ROSCommunicator(Node):
    def __init__(self):
        super().__init__('parallel_gaze_hand_drone_controller')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()
        self.data_lock = Lock()

        # 订阅和发布
        self.image_subscriber = self.create_subscription(
            ROSImage, '/yolo_detection_result', self.image_callback, qos_profile)
        self.detection_data_subscriber = self.create_subscription(
            String, '/yolo_detection_data', self.detection_data_callback, qos_profile)
        self.gaze_image_publisher = self.create_publisher(
            ROSImage, '/gaze_visualization', qos_profile)
        self.gaze_data_publisher = self.create_publisher(
            String, '/gaze_data', qos_profile)

        self.current_image = None
        self.detected_objects = []
        self.image_timestamp = None

    def image_callback(self, msg):
        try:
            with self.data_lock:
                self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.image_timestamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"图像回调错误: {e}")

    def detection_data_callback(self, msg):
        try:
            with self.data_lock:
                data = json.loads(msg.data)
                self.detected_objects = data.get('detections', [])
        except Exception as e:
            self.get_logger().error(f"检测数据回调错误: {e}")

    def publish_visualization(self, image, timestamp=None):
        try:
            ros_image = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            ros_image.header.stamp = timestamp or self.get_clock().now().to_msg()
            self.gaze_image_publisher.publish(ros_image)
        except Exception as e:
            self.get_logger().error(f"发布可视化图像错误: {e}")

    def publish_gaze_data(self, gaze_data):
        try:
            msg = String()
            msg.data = json.dumps(gaze_data, default=str)
            self.gaze_data_publisher.publish(msg)
        except Exception as e:
            self.get_logger().error(f"发布注视数据错误: {e}")

    def get_latest_data(self):
        with self.data_lock:
            return self.current_image, self.detected_objects, self.image_timestamp


class GazeEstimationThread:
    def __init__(self, gaze_data_queue: queue.Queue, control_events: Dict[str, Event]):
        self.gaze_data_queue = gaze_data_queue
        self.control_events = control_events
        self.realsense_initialized = False
        self.gaze_pipeline_initialized = False
        self.running = True
        self.screen_config = {
            'width': 2560, 'height': 1440,
            'width_mm': 596.74, 'height_mm': 335.66
        }
        self.calibrator = CameraCalibrator(self.screen_config['width'], self.screen_config['height'])
        self.calibrator.load_calibration()
        self.smoothing_config = {
            'ema_alpha': 0.3, 'outlier_threshold': 25.0, 'max_change_per_frame': 12.0,
            'motion_frames': 5, 'motion_threshold': 2.0, 'static_threshold': 0.8,
            'static_alpha': 0.05, 'motion_alpha': 0.7, 'super_static_threshold': 0.3,
            'super_static_alpha': 0.01, 'super_static_min_frames': 10, 'micro_change_threshold': 0.1
        }
        self._init_state_variables()
        self.realsense_retry_count = 0
        self.max_realsense_retries = 3
        self.realsense_retry_delay = 2.0

    def _init_state_variables(self):
        self.smoothed = {'pitch': None, 'yaw': None}
        self.final_output = {'pitch': None, 'yaw': None}
        self.raw_history = deque(maxlen=self.smoothing_config['motion_frames'])
        self.motion_state = "unknown"
        self.static_frames = 0
        self.super_static_frames = 0
        self.gaze_point_history = deque(maxlen=8)
        self.stable_gaze_point = None
        self.camera_calibration_mode = False
        self.current_calibration_point = 0
        self.calibration_data_collected = False
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

    def initialize_realsense_with_retry(self):
        for attempt in range(self.max_realsense_retries):
            try:
                print(f"RealSense初始化尝试 {attempt + 1}/{self.max_realsense_retries}")
                time.sleep(self.realsense_retry_delay * attempt)

                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

                profile = self.pipeline.start(config)
                self.color_intrinsics = rs.video_stream_profile(
                    profile.get_stream(rs.stream.color)).get_intrinsics()
                self.align = rs.align(rs.stream.color)

                for _ in range(10):
                    frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                    if not frames:
                        raise Exception("预热时无法获取帧")

                self.realsense_initialized = True
                print("RealSense D435 初始化成功")
                return True

            except Exception as e:
                print(f"RealSense初始化失败 (尝试 {attempt + 1}): {e}")
                if hasattr(self, 'pipeline'):
                    try:
                        self.pipeline.stop()
                    except:
                        pass
                    delattr(self, 'pipeline')

                if attempt < self.max_realsense_retries - 1:
                    time.sleep(self.realsense_retry_delay)

        print("RealSense初始化最终失败，将以模拟模式运行")
        self.realsense_initialized = False
        return False

    def initialize(self):
        try:
            model_path = pathlib.Path.cwd() / 'models' / 'L2CSNet_gaze360.pkl'
            if not model_path.exists():
                raise FileNotFoundError(f"模型文件不存在: {model_path}")

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"使用{'GPU' if device.type == 'cuda' else 'CPU'}加速")
            self.gaze_pipeline = Pipeline(weights=model_path, arch='ResNet50', device=device)
            self.gaze_pipeline_initialized = True

            self.initialize_realsense_with_retry()
            return True

        except Exception as e:
            print(f"初始化失败: {e}")
            return False

    def get_eye_position_3d(self, bbox, depth_frame):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            eye_x = max(0, min(639, (x1 + x2) // 2))
            eye_y = max(0, min(479, y1 + int((y2 - y1) * 0.3)))

            depth = depth_frame.get_distance(eye_x, eye_y)
            if depth == 0:
                return None

            eye_3d_m = rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [eye_x, eye_y], depth)
            return [coord * 1000 for coord in eye_3d_m]
        except:
            return None

    def detect_motion_state(self, pitch, yaw):
        self.raw_history.append((pitch, yaw))
        if len(self.raw_history) < self.smoothing_config['motion_frames']:
            return "unknown"

        history = list(self.raw_history)
        changes = [math.hypot(history[i][0] - history[i - 1][0], history[i][1] - history[i - 1][1])
                   for i in range(1, len(history))]
        avg_change, max_change = np.mean(changes), np.max(changes)

        if max_change > self.smoothing_config['motion_threshold'] or avg_change > self.smoothing_config[
            'static_threshold']:
            self.static_frames = 0
            self.super_static_frames = 0
            return "moving"
        else:
            self.static_frames += 1
            if self.static_frames > 3:
                if (max_change < self.smoothing_config['super_static_threshold'] and
                        avg_change < self.smoothing_config['super_static_threshold'] * 0.5):
                    self.super_static_frames += 1
                    if self.super_static_frames >= self.smoothing_config['super_static_min_frames']:
                        return "super_static"
                return "static"
            return "moving"

    def smooth_angles(self, raw_pitch, raw_yaw):
        if self.smoothed['pitch'] is None:
            self.smoothed['pitch'] = self.final_output['pitch'] = raw_pitch
            self.smoothed['yaw'] = self.final_output['yaw'] = raw_yaw
            return raw_pitch, raw_yaw

        if math.hypot(raw_pitch - self.final_output['pitch'],
                      raw_yaw - self.final_output['yaw']) > self.smoothing_config['outlier_threshold']:
            return self.final_output['pitch'], self.final_output['yaw']

        for axis, raw_val in [('pitch', raw_pitch), ('yaw', raw_yaw)]:
            diff = raw_val - self.smoothed[axis]
            if abs(diff) > self.smoothing_config['max_change_per_frame']:
                raw_val = self.smoothed[axis] + math.copysign(self.smoothing_config['max_change_per_frame'], diff)
            self.smoothed[axis] = (self.smoothing_config['ema_alpha'] * raw_val +
                                   (1 - self.smoothing_config['ema_alpha']) * self.smoothed[axis])

        self.motion_state = self.detect_motion_state(self.smoothed['pitch'], self.smoothed['yaw'])
        current_change = math.hypot(self.smoothed['pitch'] - self.final_output['pitch'],
                                    self.smoothed['yaw'] - self.final_output['yaw'])
        if (self.motion_state in ["static", "super_static"] and
                current_change < self.smoothing_config['micro_change_threshold']):
            return self.final_output['pitch'], self.final_output['yaw']

        alpha_map = {
            "moving": self.smoothing_config['motion_alpha'],
            "super_static": self.smoothing_config['super_static_alpha'],
            "static": self.smoothing_config['static_alpha'] * (0.2 if current_change < 0.5 else 1)
        }
        alpha = alpha_map.get(self.motion_state, self.smoothing_config['static_alpha'])

        for axis in ['pitch', 'yaw']:
            self.final_output[axis] = alpha * self.smoothed[axis] + (1 - alpha) * self.final_output[axis]

        return self.final_output['pitch'], self.final_output['yaw']

    def calculate_gaze_intersection(self, eye_pos_3d, pitch_deg, yaw_deg):
        if not eye_pos_3d:
            return None

        try:
            eye_x, eye_y, eye_z = eye_pos_3d
            camera_screen_y = -self.screen_config['height_mm'] / 2 + 70
            screen_eye_x = eye_x
            screen_eye_y = -eye_y + camera_screen_y
            screen_eye_z = eye_z

            pitch_rad, yaw_rad = math.radians(pitch_deg), math.radians(yaw_deg)
            gaze_x = math.sin(yaw_rad)
            gaze_y = math.sin(pitch_rad)
            gaze_z = -math.cos(pitch_rad) * math.cos(yaw_rad)

            if abs(gaze_z) < 1e-6:
                return None

            t = -screen_eye_z / gaze_z
            if t < 0:
                return None

            intersection_x = screen_eye_x + t * gaze_x
            intersection_y = screen_eye_y + t * gaze_y

            pixel_x = (intersection_x + self.screen_config['width_mm'] / 2) * self.screen_config['width'] / \
                      self.screen_config['width_mm']
            pixel_y = (self.screen_config['height_mm'] / 2 - intersection_y) * self.screen_config['height'] / \
                      self.screen_config['height_mm']

            return (pixel_x, pixel_y)
        except:
            return None

    def smooth_gaze_point(self, gaze_point):
        if not gaze_point:
            return self.stable_gaze_point

        self.gaze_point_history.append(gaze_point)
        if len(self.gaze_point_history) < 3:
            self.stable_gaze_point = gaze_point
            return gaze_point

        if self.motion_state == "moving":
            recent_points = list(self.gaze_point_history)[-3:]
            x_coords, y_coords = zip(*recent_points)
            smoothed_point = (np.mean(x_coords), np.mean(y_coords))
        elif self.motion_state == "super_static":
            x_coords, y_coords = zip(*self.gaze_point_history)
            weights = np.linspace(0.5, 1.0, len(x_coords))
            weights = weights / np.sum(weights)
            smoothed_point = (np.average(x_coords, weights=weights), np.average(y_coords, weights=weights))

            if (self.stable_gaze_point and
                    math.hypot(smoothed_point[0] - self.stable_gaze_point[0],
                               smoothed_point[1] - self.stable_gaze_point[1]) < 5):
                return self.stable_gaze_point
        else:
            recent_points = list(self.gaze_point_history)[-5:]
            x_coords, y_coords = zip(*recent_points)
            smoothed_point = (np.mean(x_coords), np.mean(y_coords))

        self.stable_gaze_point = smoothed_point
        return smoothed_point

    def handle_camera_calibration(self, frame, results, depth_frame):
        if not self.camera_calibration_mode:
            return

        calibration_points = self.calibrator.get_calibration_points()

        if self.calibration_data_collected and results and len(results.bboxes) > 0:
            best_idx = results.scores.argmax()
            raw_pitch = math.degrees(results.yaw[best_idx])
            raw_yaw = math.degrees(results.pitch[best_idx])
            bbox = results.bboxes[best_idx]
            eye_pos_3d = self.get_eye_position_3d(bbox, depth_frame)

            if eye_pos_3d:
                current_point = calibration_points[self.current_calibration_point]
                self.calibrator.add_calibration_data(current_point, raw_pitch, raw_yaw, eye_pos_3d)
                print(f"已收集校准点 {self.current_calibration_point + 1}: {current_point}")
                self.current_calibration_point += 1

                if self.current_calibration_point >= len(calibration_points):
                    if self.calibrator.calculate_camera_pose():
                        print("摄像机校准完成！")
                        self.camera_calibration_mode = False
                    else:
                        self.reset_camera_calibration()

            self.calibration_data_collected = False

    def reset_camera_calibration(self):
        self.current_calibration_point = 0
        self.calibration_data_collected = False
        self.calibrator.reset_calibration()

    def calculate_fps(self):
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            self.current_fps = 30 / elapsed_time
            self.fps_start_time = current_time

    def reset_state(self):
        self._init_state_variables()

    def run(self):
        print("开始初始化...")
        if not self.initialize():
            print("初始化失败，线程退出")
            return

        print("初始化完成，开始运行")

        try:
            while self.running and not self.control_events['exit'].is_set():
                try:
                    self._handle_control_events()

                    gaze_data = self._create_base_gaze_data()

                    if self.realsense_initialized:
                        self._process_realsense_data(gaze_data)
                    else:
                        time.sleep(0.033)

                    self._queue_gaze_data(gaze_data)

                except Exception as e:
                    print(f"主循环错误: {e}")
                    time.sleep(0.01)

        finally:
            self.cleanup()

    def _handle_control_events(self):
        if self.control_events['reset'].is_set():
            self.reset_state()
            self.control_events['reset'].clear()

        if self.control_events['calibration_mode'].is_set():
            if not self.camera_calibration_mode:
                self.camera_calibration_mode = True
                self.reset_camera_calibration()
            self.control_events['calibration_mode'].clear()

        if self.control_events['calibration_data'].is_set():
            if self.camera_calibration_mode:
                self.calibration_data_collected = True
            self.control_events['calibration_data'].clear()

    def _create_base_gaze_data(self):
        return GazeData(
            timestamp=time.time(),
            gaze_point=None,
            img_gaze_point=None,
            gazed_object=None,
            eye_pos_3d=None,
            motion_state=self.motion_state,
            smoothed_pitch=0.0,
            smoothed_yaw=0.0,
            raw_pitch=0.0,
            raw_yaw=0.0,
            corrected_pitch=0.0,
            corrected_yaw=0.0,
            face_bbox=None,
            face_score=0.0,
            tracking_status="无目标",
            yaw_control_mode="gaze",
            current_yaw_speed=0.0,
            calibration_mode=self.camera_calibration_mode,
            realsense_connected=self.realsense_initialized
        )

    def _process_realsense_data(self, gaze_data):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            if not frames:
                return

            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return

            frame = np.asanyarray(color_frame.get_data())
            results = self.gaze_pipeline.step(frame)

            self.calculate_fps()
            self.handle_camera_calibration(frame, results, depth_frame)

            if results and len(results.bboxes) > 0:
                self._process_gaze_results(results, depth_frame, gaze_data)

        except Exception as e:
            print(f"RealSense处理错误: {e}")

    def _process_gaze_results(self, results, depth_frame, gaze_data):
        try:
            best_idx = results.scores.argmax()
            bbox = results.bboxes[best_idx]
            raw_pitch, raw_yaw = math.degrees(results.yaw[best_idx]), math.degrees(results.pitch[best_idx])

            corrected_pitch, corrected_yaw = self.calibrator.correct_angles(raw_pitch, raw_yaw)
            smoothed_pitch, smoothed_yaw = self.smooth_angles(corrected_pitch, corrected_yaw)
            eye_pos_3d = self.get_eye_position_3d(bbox, depth_frame)
            gaze_point = self.calculate_gaze_intersection(eye_pos_3d, smoothed_pitch, smoothed_yaw)

            if gaze_point:
                gaze_point = self.smooth_gaze_point(gaze_point)

            gaze_data.gaze_point = gaze_point
            gaze_data.eye_pos_3d = eye_pos_3d
            gaze_data.smoothed_pitch = smoothed_pitch
            gaze_data.smoothed_yaw = smoothed_yaw
            gaze_data.raw_pitch = raw_pitch
            gaze_data.raw_yaw = raw_yaw
            gaze_data.corrected_pitch = corrected_pitch
            gaze_data.corrected_yaw = corrected_yaw
            gaze_data.face_bbox = tuple(bbox)
            gaze_data.face_score = float(results.scores[best_idx])

        except Exception as e:
            print(f"处理结果错误: {e}")

    def _queue_gaze_data(self, gaze_data):
        try:
            self.gaze_data_queue.put_nowait(gaze_data)
        except queue.Full:
            try:
                self.gaze_data_queue.get_nowait()
                self.gaze_data_queue.put_nowait(gaze_data)
            except queue.Empty:
                pass

    def cleanup(self):
        try:
            if hasattr(self, 'pipeline') and self.realsense_initialized:
                self.pipeline.stop()
                print("RealSense pipeline已停止")
        except Exception as e:
            print(f"清理资源时出错: {e}")

    def stop(self):
        self.running = False

class ParallelIntegratedGazeDroneController:
    def __init__(self, drone):
        print("初始化修复版并行优化集成控制系统...")

        self.drone = drone
        self.drone_ready = True
        self.control_active = True
        self.ros_initialized = False

        self._init_ros_communication()

        # 屏幕参数
        self.screen_config = {
            'width': 2560, 'height': 1440,
            'width_mm': 596.74, 'height_mm': 335.66,
            'center_x': 1280, 'yaw_deadzone': 200
        }

        self._init_threading_components()

        self._init_components()

        self._init_gesture_control()

        self._init_control_parameters()

        print("修复版并行优化集成眼神追踪+手势控制+无人机系统初始化完成")

    def _init_ros_communication(self):
        try:
            rclpy.init()
            self.ros_comm = ROSCommunicator()
            self.ros_initialized = True
            self.ros_thread = threading.Thread(target=self.ros_spin_thread, daemon=True)
            self.ros_thread.start()
        except Exception as e:
            print(f"ROS2初始化失败: {e}")

    def _init_threading_components(self):
        self.gaze_data_queue = queue.Queue(maxsize=10)
        self.control_events = {
            'exit': Event(),
            'reset': Event(),
            'calibration_mode': Event(),
            'calibration_data': Event(),
            'instant_mode_toggle': Event()
        }

    def _init_components(self):
        self.gaze_matcher = InstantGazeObjectMatcher(history_size=10, dwell_threshold=5, margin=20)
        self.gaze_matcher.set_fast_mode(True)

        self.object_tracker = ObjectTracker(lock_threshold_frames=20, lock_threshold_seconds=0.6)

        self.gaze_estimation_thread = GazeEstimationThread(self.gaze_data_queue, self.control_events)

        self.latest_gaze_data = None
        self.current_yaw_speed = 0.0
        self.yaw_control_mode = "gaze"

    def _init_gesture_control(self):
        try:
            self.gesture_model = GestureModelWrapper(MODEL_FILE)
            self.gesture_votes = deque(maxlen=max(1, SMOOTH_K))
            print("手势识别模型加载成功")
        except Exception as e:
            print(f"手势识别模型加载失败: {e}")
            self.gesture_model = None
            self.gesture_votes = deque(maxlen=5)

        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.drone_state = {
            'taken_off': False,
            'trajectory_mode': False,
            'current_gesture': 'noges',
            'last_command': None
        }

        self.trajectory_3d = []
        self.trajectory_log = []
        self.recording_trajectory = False
        self.last_gesture_time = time.time()
        self.gesture_cooldown = 0.5
        self.trajectory_index = 0

        self._init_kalman_filter()

        self.speed_multiplier = 1.0
        self.max_speed_multiplier = 2
        self.min_speed_multiplier = 0.3
        self.speed_change_rate = 0.05

        self.current_errors = [0.0, 0.0, 0.0]
        self.control_mode = 'idle'
        self.last_valid_errors = [0.0, 0.0, 0.0]
        self.last_trajectory_errors = [0.0, 0.0, 0.0]
    def _init_kalman_filter(self):
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.F = np.array([[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
        self.kf.P *= 1000
        self.kf.R *= 0.01
        self.kf.Q *= 1e-4
        self.first_point = True

    def _init_control_parameters(self):

        self.yaw_speed_scale = 0.05
        self.max_yaw_speed = 30.0

        self.death = 0.005
        self.vnorth_last = 0.0
        self.veast_last = 0.0
        self.vdown_last = 0.0

        self.MIN_DEPTH = 0.1
        self.MAX_DEPTH = 1.0
        self.colormap = cm.get_cmap("jet")

        self.image_frame_count = 0
        self.image_fps_start_time = time.time()
        self.current_image_fps = 0

    def ros_spin_thread(self):
        try:
            rclpy.spin(self.ros_comm)
        except Exception as e:
            print(f"ROS2线程错误: {e}")

    def convert_gaze_to_image_coordinates(self, gaze_point, image_width, image_height):
        if not gaze_point:
            return None

        try:
            screen_x, screen_y = gaze_point
            img_x = screen_x * image_width / self.screen_config['width']
            img_y = screen_y * image_height / self.screen_config['height']
            img_x = max(0, min(image_width - 1, img_x))
            img_y = max(0, min(image_height - 1, img_y))
            return (img_x, img_y)
        except Exception as e:
            print(f"坐标转换错误: {e}")
            return None

    def calculate_yaw_speed_from_gaze(self, gaze_point):
        if self.latest_gaze_data and self.latest_gaze_data.calibration_mode:
            return 0.0

        if not gaze_point:
            return 0.0

        gaze_x, gaze_y = gaze_point
        x_offset = gaze_x - self.screen_config['center_x']

        if abs(x_offset) <= self.screen_config['yaw_deadzone']:
            return 0.0

        effective_offset = (x_offset - self.screen_config['yaw_deadzone'] if x_offset > 0
                            else x_offset + self.screen_config['yaw_deadzone'])
        yaw_speed = effective_offset * self.yaw_speed_scale
        return np.clip(yaw_speed, -self.max_yaw_speed, self.max_yaw_speed)

    def update_yaw_control(self, gaze_point, gazed_object, detected_objects, frame_width):
        if self.latest_gaze_data and self.latest_gaze_data.calibration_mode:
            self.current_yaw_speed = 0.0
            self.yaw_control_mode = "disabled"
            return

        self.object_tracker.update(gazed_object, detected_objects, frame_width)

        if self.object_tracker.is_locked:
            self.current_yaw_speed = self.object_tracker.calculate_tracking_yaw_speed(frame_width)
            self.yaw_control_mode = "tracking"
        else:
            self.current_yaw_speed = self.calculate_yaw_speed_from_gaze(gaze_point)
            self.yaw_control_mode = "gaze"

    def draw_gaze_visualization(self, image, gaze_data: GazeData, detected_objects):
        vis_image = image.copy()
        height, width = vis_image.shape[:2]

        img_gaze_point = None
        if gaze_data.gaze_point:
            img_gaze_point = self.convert_gaze_to_image_coordinates(gaze_data.gaze_point, width, height)
            gaze_data.img_gaze_point = img_gaze_point

        gazed_object = self.gaze_matcher.find_gazed_object(img_gaze_point, detected_objects)

        self.update_yaw_control(gaze_data.gaze_point, gazed_object, detected_objects, width)

        gaze_data.tracking_status = self.object_tracker.get_status()
        gaze_data.yaw_control_mode = self.yaw_control_mode
        gaze_data.current_yaw_speed = self.current_yaw_speed
        gaze_data.gazed_object = gazed_object

        self._draw_calibration_mode(vis_image, gaze_data, width, height)
        self._draw_gaze_point(vis_image, gaze_data, img_gaze_point, width, height)
        self._draw_gazed_objects(vis_image, gaze_data, gazed_object, detected_objects, width, height)
        self._draw_status_info(vis_image, gaze_data, detected_objects)

        return vis_image, gazed_object

    def _draw_calibration_mode(self, vis_image, gaze_data, width, height):
        if not gaze_data.calibration_mode:
            return

        calibration_points = self.gaze_estimation_thread.calibrator.get_calibration_points()
        current_point_idx = self.gaze_estimation_thread.current_calibration_point

        for i, (screen_x, screen_y) in enumerate(calibration_points):
            img_x = int(screen_x * width / self.screen_config['width'])
            img_y = int(screen_y * height / self.screen_config['height'])
            img_x = max(30, min(width - 30, img_x))
            img_y = max(30, min(height - 30, img_y))

            if i == current_point_idx:
                cv2.circle(vis_image, (img_x, img_y), 40, (0, 0, 255), 3)
                cv2.circle(vis_image, (img_x, img_y), 20, (0, 0, 255), 2)
                cv2.circle(vis_image, (img_x, img_y), 10, (0, 0, 255), -1)
                cv2.putText(vis_image, str(i + 1), (img_x - 15, img_y + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                if int(time.time() * 2) % 2:
                    cv2.circle(vis_image, (img_x, img_y), 50, (0, 255, 255), 2)
            elif i < current_point_idx:
                cv2.circle(vis_image, (img_x, img_y), 20, (0, 255, 0), 2)
                cv2.circle(vis_image, (img_x, img_y), 10, (0, 255, 0), -1)
                cv2.putText(vis_image, str(i + 1), (img_x - 8, img_y + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(vis_image, "✓", (img_x + 25, img_y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.circle(vis_image, (img_x, img_y), 15, (128, 128, 128), 2)
                cv2.circle(vis_image, (img_x, img_y), 5, (128, 128, 128), -1)

        info_texts = [
            f"标定进度: {current_point_idx}/9",
            f"请注视红色标定点 {current_point_idx + 1}",
            "注视稳定后按空格键捕获数据",
            "标定期间无人机航向控制已禁用",
            f"RealSense: {'已连接' if gaze_data.realsense_connected else '未连接'}"
        ]

        cv2.rectangle(vis_image, (10, height - 160), (650, height - 20), (0, 0, 0), -1)
        cv2.rectangle(vis_image, (10, height - 160), (650, height - 20), (0, 255, 255), 2)

        for i, text in enumerate(info_texts):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            if i == 1:
                color = (0, 0, 255)
            elif "已连接" in text:
                color = (0, 255, 0)
            elif "未连接" in text:
                color = (0, 0, 255)
            cv2.putText(vis_image, text, (20, height - 130 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_gaze_point(self, vis_image, gaze_data, img_gaze_point, width, height):
        if gaze_data.calibration_mode and img_gaze_point:
            x, y = int(img_gaze_point[0]), int(img_gaze_point[1])
            x, y = max(0, min(width - 1, x)), max(0, min(height - 1, y))
            cv2.circle(vis_image, (x, y), 15, (0, 255, 0), 3)
            cv2.circle(vis_image, (x, y), 2, (0, 255, 0), -1)
        elif not gaze_data.calibration_mode and img_gaze_point:
            x, y = int(img_gaze_point[0]), int(img_gaze_point[1])
            x, y = max(0, min(width - 1, x)), max(0, min(height - 1, y))
            cv2.circle(vis_image, (x, y), 15, (0, 255, 0), 3)
            cv2.circle(vis_image, (x, y), 2, (0, 255, 0), -1)
            if gaze_data.gaze_point:
                coord_text = f"Screen:({gaze_data.gaze_point[0]:.0f},{gaze_data.gaze_point[1]:.0f})"
                cv2.putText(vis_image, coord_text, (x + 25, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif not img_gaze_point:
            status_text = "No gaze point detected"
            if not gaze_data.realsense_connected:
                status_text = "RealSense not connected - check device"
            cv2.putText(vis_image, status_text, (10, height - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def _draw_gazed_objects(self, vis_image, gaze_data, gazed_object, detected_objects, width, height):
        if gaze_data.calibration_mode:
            return

        if gazed_object:
            self._highlight_object(vis_image, gazed_object)
        elif self.object_tracker.is_locked and self.object_tracker.current_target:
            locked_obj = self._find_locked_object(detected_objects)
            if locked_obj:
                self._highlight_locked_object(vis_image, locked_obj)

    def _highlight_object(self, vis_image, gazed_object):
        x1, y1, x2, y2 = [int(coord) for coord in gazed_object['bbox']]

        if self.object_tracker.is_locked and self._is_same_object_for_display(self.object_tracker.current_target,
                                                                              gazed_object):
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
            if int(time.time() * 4) % 2:
                cv2.rectangle(vis_image, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (255, 255, 0), 3)
            label = f"LOCKED: {gazed_object['class_name']} ({gazed_object['confidence']:.2f})"
            self._draw_object_label(vis_image, label, x1, y1, (0, 0, 255))
        elif hasattr(self.object_tracker, 'new_target_candidate') and self._is_same_object_for_display(
                self.object_tracker.new_target_candidate, gazed_object):
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 165, 255), 3)
            label = f"CANDIDATE: {gazed_object['class_name']} ({gazed_object['confidence']:.2f})"
            self._draw_object_label(vis_image, label, x1, y1, (0, 165, 255))
        else:
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 3)
            label = f"GAZING: {gazed_object['class_name']} ({gazed_object['confidence']:.2f})"
            self._draw_object_label(vis_image, label, x1, y1, (0, 255, 255))

    def _draw_object_label(self, vis_image, label, x, y, color):
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(vis_image, (x, y - 35), (x + label_size[0], y), color, -1)
        cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _find_locked_object(self, detected_objects):
        for obj in detected_objects:
            if self._is_same_object_for_display(self.object_tracker.current_target, obj):
                return obj
        return None

    def _highlight_locked_object(self, vis_image, locked_obj):
        x1, y1, x2, y2 = [int(coord) for coord in locked_obj['bbox']]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 150), 4)
        label = f"LOCKED (No Gaze): {locked_obj['class_name']} ({locked_obj['confidence']:.2f})"
        self._draw_object_label(vis_image, label, x1, y1, (0, 0, 150))

    def _is_same_object_for_display(self, obj1, obj2):
        if not obj1 or not obj2 or obj1['class_name'] != obj2['class_name']:
            return False
        center1 = ((obj1['bbox'][0] + obj1['bbox'][2]) / 2, (obj1['bbox'][1] + obj1['bbox'][3]) / 2)
        center2 = ((obj2['bbox'][0] + obj2['bbox'][2]) / 2, (obj2['bbox'][1] + obj2['bbox'][3]) / 2)
        distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance < 100

    def _draw_status_info(self, vis_image, gaze_data, detected_objects):
        status_items = [
            f"Face Detection: {'✓' if gaze_data.gaze_point else '✗'}",
            f"Gaze FPS: {self.gaze_estimation_thread.current_fps:.1f}",
            f"Image FPS: {self.current_image_fps:.1f}",
        ]

        for i, status in enumerate(status_items):
            color = (0, 255, 0) if ('ON' in status or '✓' in status or '🎯' in status) else (0, 255, 255)
            if '✗' in status:
                color = (0, 0, 255)
            elif 'Control Mode:' in status:
                mode_colors = {
                    'TRAJECTORY': (0, 255, 0),
                    'THUMB': (0, 255, 255),
                    'FIST': (255, 0, 255),
                    'NOGES': (128, 128, 128),
                    'BRAKE': (0, 0, 255),
                    'IDLE': (255, 255, 255)
                }
                color = mode_colors.get(self.control_mode.upper(), (255, 255, 255))
            cv2.putText(vis_image, status, (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def calculate_image_fps(self):
        self.image_frame_count += 1
        if self.image_frame_count % 30 == 0:
            current_time = time.time()
            elapsed_time = current_time - self.image_fps_start_time
            self.current_image_fps = 30 / elapsed_time
            self.image_fps_start_time = current_time

    def detect_gesture(self, hand_landmarks):
        if not self.gesture_model:
            return 'noges', 0.0

        try:
            feat = self.gesture_model.extract_landmarks(hand_landmarks)
            probs = self.gesture_model.predict(feat)
            idx = int(np.argmax(probs))
            max_prob = float(probs[idx])

            self.gesture_votes.append(idx)
            if len(self.gesture_votes) >= 3:
                pred_idx = max(set(self.gesture_votes), key=self.gesture_votes.count)
                pred_prob = float(probs[pred_idx])
            else:
                pred_idx = idx
                pred_prob = max_prob

            gesture_name = self.gesture_model.id2label.get(pred_idx, 'noges')

            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0

            if self._debug_counter % 30 == 0:
                print(f"手势概率分布: {dict(zip(self.gesture_model.id2label.values(), [f'{p:.3f}' for p in probs]))}")
                print(f"投票结果: {gesture_name} (置信度: {pred_prob:.3f})")

            return gesture_name, pred_prob

        except Exception as e:
            print(f"手势识别错误: {e}")
            import traceback
            traceback.print_exc()
            return 'noges', 0.0

    def update_speed_multiplier(self, gesture):
        if gesture == 'thumb':
            self.speed_multiplier = min(
                self.max_speed_multiplier,
                self.speed_multiplier + self.speed_change_rate
            )
        elif gesture == 'fist':
            self.speed_multiplier = max(
                self.min_speed_multiplier,
                self.speed_multiplier - self.speed_change_rate
            )
        elif gesture == 'brake':
            self.speed_multiplier = 0.0

    async def _send_unified_control_command(self):
        if not self.drone_state['taken_off']:
            return

        errors = self.current_errors.copy()

        for i in range(3):
            if abs(errors[i]) < self.death:
                errors[i] = 0

        adjusted_errors = [error * self.speed_multiplier for error in errors]

        vnorth_current = -adjusted_errors[2] * 100
        veast_current = -adjusted_errors[0] * 100
        vdown_current = adjusted_errors[1] * 100

        await self.send_drone_command(vnorth_current, veast_current, vdown_current, self.current_yaw_speed)

        self.vnorth_last = vnorth_current
        self.veast_last = veast_current
        self.vdown_last = vdown_current

    def is_valid_point(self, pt):
        return all(not math.isnan(c) and not math.isinf(c) for c in pt)

    def is_outlier(self, new_pt, prev_pt, threshold=0.15):
        return np.linalg.norm(np.array(new_pt) - np.array(prev_pt)) > threshold

    async def send_drone_command(self, vnorth, veast, vdown, yaw_speed):
        if self.drone and self.drone_ready:
            try:
                print(f"vnorth={vnorth},veast={veast},vdown={vdown}")
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(vnorth, veast, vdown, yaw_speed)
                )
                return True
            except Exception as e:
                print(f"发送无人机命令失败: {e}")
                return False
        return False

    async def process_hand_gesture_with_realsense(self):
        if not self.gaze_estimation_thread.realsense_initialized:
            return None

        try:
            frames = self.gaze_estimation_thread.pipeline.wait_for_frames(timeout_ms=100)
            if not frames:
                return None

            aligned_frames = self.gaze_estimation_thread.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None

            frame = np.asanyarray(color_frame.get_data())
            return await self.process_hand_gesture(frame, depth_frame)
        except:
            return None

    async def process_hand_gesture(self, frame, depth_frame):
        current_time = time.time()
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = self.mp_hands.process(rgb_img)

        if hand_result.multi_hand_landmarks:
            hand_landmarks = hand_result.multi_hand_landmarks[0]

            try:
                await self._handle_gesture_detection(hand_landmarks, current_time, depth_frame)
                if self.drone_state['current_gesture'] != 'noges':
                    await self._handle_trajectory_recording(hand_landmarks.landmark, current_time, depth_frame)

                self._draw_hand_visualization(frame, hand_result, hand_landmarks.landmark)

            except Exception as e:
                print(f"手势处理错误: {e}")

        else:
            self.drone_state['current_gesture'] = 'noges'
            await self._handle_no_gesture()

        return frame

    async def _handle_gesture_detection(self, hand_landmarks, current_time, depth_frame):
        try:
            gesture, confidence = self.detect_gesture(hand_landmarks)

            previous_gesture = self.drone_state['current_gesture']
            self.drone_state['current_gesture'] = gesture

            if confidence < 0.6:
                return

            if gesture == '1':
                await self._handle_drawing_gesture(previous_gesture, current_time)
                return

            if gesture in ['ok', 'brake'] and (current_time - self.last_gesture_time <= self.gesture_cooldown):
                return

            print(f"检测到手势: {gesture} (置信度: {confidence:.3f})")

            if gesture == 'ok' and not self.drone_state['taken_off']:
                await self._handle_takeoff_gesture(current_time)

            elif gesture == 'fist':
                await self._handle_deceleration_gesture()

            elif gesture == 'thumb':
                await self._handle_acceleration_gesture()

            elif gesture == 'brake':
                await self._handle_brake_gesture()
                self.last_gesture_time = current_time

            elif gesture == 'noges':
                await self._handle_no_gesture()

            if previous_gesture == '1' and gesture != '1':
                await self._stop_drawing_but_keep_trajectory_mode()

        except Exception as e:
            print(f"手势处理错误详细信息: {e}")
            import traceback
            traceback.print_exc()

    async def _handle_takeoff_gesture(self, current_time):
        if not self.drone_state['taken_off']:
            success = await takeoff_drone(self.drone)
            if success:
                self.drone_state['taken_off'] = True
                print("无人机起飞完成，可以使用其他手势控制")
            self.last_gesture_time = current_time

    async def _handle_drawing_gesture(self, previous_gesture, current_time):
        if not self.drone_state['taken_off']:
            return

        if previous_gesture != '1':
            if (current_time - self.last_gesture_time <= self.gesture_cooldown):
                return

            if self.control_mode != 'trajectory':
                self.current_errors = self.last_valid_errors.copy()
                print(f"进入手势1模式，保持当前errors: {self.current_errors}")

            self.recording_trajectory = True
            self.drone_state['trajectory_mode'] = True
            self.trajectory_3d.clear()
            self.trajectory_log.clear()
            self.first_point = True
            self.trajectory_index = 0
            self.last_trajectory_errors = [0.0, 0.0, 0.0]
            self.control_mode = 'trajectory'
            self.speed_multiplier = 1.0
            self.last_gesture_time = current_time
            print("开始绘制新轨迹 - 保持手势1继续绘制，速度倍数已重置为1.0")

    async def _stop_drawing_but_keep_trajectory_mode(self):
        if self.recording_trajectory:
            self.recording_trajectory = False
            print(f"停止绘制轨迹，已记录 {len(self.trajectory_3d)} 个轨迹点")
            print("可以使用THUMB/FIST调整速度，或再次做手势1开始新轨迹")

    async def _handle_acceleration_gesture(self):
        if self.control_mode != 'thumb':
            self.current_errors = self.last_valid_errors.copy()
            self.control_mode = 'thumb'
            print(f"进入THUMB模式，复制当前errors: {self.current_errors}")

        self.update_speed_multiplier('thumb')
        await self._send_unified_control_command()

        print(f"加速，当前速度倍数: {self.speed_multiplier:.2f}")

    async def _handle_deceleration_gesture(self):
        if self.control_mode != 'fist':
            self.current_errors = self.last_valid_errors.copy()
            self.control_mode = 'fist'
            print(f"进入FIST模式，复制当前errors: {self.current_errors}")
        self.update_speed_multiplier('fist')

        await self._send_unified_control_command()

        print(f"减速，当前速度倍数: {self.speed_multiplier:.2f}")

    async def _handle_brake_gesture(self):
        self.current_errors = [0.0, 0.0, 0.0]
        self.last_valid_errors = [0.0, 0.0, 0.0]
        self.control_mode = 'brake'
        self.last_trajectory_errors = [0.0, 0.0, 0.0]
        self.update_speed_multiplier('brake')
        await self.send_drone_command(0, 0, 0, 0)
        self.vnorth_last = self.veast_last = self.vdown_last = 0.0

        self.trajectory_3d.clear()
        self.trajectory_log.clear()
        self.trajectory_index = 0
        self.recording_trajectory = False
        self.drone_state['trajectory_mode'] = False
        self.first_point = True

        print("紧急刹车，errors归零，轨迹已清空")

    async def _handle_no_gesture(self):
        if self.control_mode != 'noges':
            self.current_errors = self.last_valid_errors.copy()
            self.control_mode = 'noges'
            print(f"进入NOGES模式，保持当前errors: {self.current_errors}")

        await self._send_unified_control_command()

    def _start_trajectory_recording(self, current_time):
        self.recording_trajectory = True
        self.trajectory_3d.clear()
        self.trajectory_log.clear()
        self.first_point = True
        self.trajectory_index = 0
        self.last_gesture_time = current_time
        print("检测到OK手势，开始记录轨迹！")

    async def _stop_trajectory_recording(self, current_time):
        self.trajectory_index = 0
        await self.send_drone_command(0, 0, 0, 0)
        self.trajectory_3d.clear()
        self.trajectory_log.clear()
        self.recording_trajectory = False
        self.first_point = True
        self.last_gesture_time = current_time
        self.vnorth_last = self.veast_last = self.vdown_last = 0.0

    async def _handle_trajectory_recording(self, landmarks, current_time, depth_frame):
        if not self.recording_trajectory:
            return

        lm = landmarks[5]
        u = int(lm.x * self.gaze_estimation_thread.color_intrinsics.width)
        v = int(lm.y * self.gaze_estimation_thread.color_intrinsics.height)
        depth = depth_frame.get_distance(u, v)
        x, y, z = rs.rs2_deproject_pixel_to_point(self.gaze_estimation_thread.color_intrinsics, [u, v], depth)

        if self.is_valid_point([x, y, z]):
            if self.trajectory_3d and self.is_outlier([x, y, z], self.trajectory_3d[-1]):
                return

            if self.first_point:
                self.kf.x[:3] = np.array([[x], [y], [z]])
                self.kf.x[3:] = 0
                self.first_point = False
            else:
                self.kf.predict()
                self.kf.update(np.array([[x], [y], [z]]))
                x, y, z = self.kf.x[:3, 0]

            self.trajectory_3d.append([x, y, z])
            self.trajectory_log.append({'x': x, 'y': y, 'z': z, 'timestamp': current_time})
            self.trajectory_index += 1

            if self.trajectory_index > 5:
                await self._send_trajectory_control()

    async def _send_trajectory_control(self):
        recent_indices = list(range(max(0, self.trajectory_index - 4), self.trajectory_index))
        prev_indices = list(range(max(0, self.trajectory_index - 7), self.trajectory_index - 3))

        if len(recent_indices) < 3 or len(prev_indices) < 3:
            return

        current_pos = [np.mean([self.trajectory_3d[i][j] for i in recent_indices]) for j in range(3)]
        prev_pos = [np.mean([self.trajectory_3d[i][j] for i in prev_indices]) for j in range(3)]

        errors = [current_pos[i] - prev_pos[i] for i in range(3)]

        total_error = math.sqrt(sum(e ** 2 for e in errors))
        if total_error < 0.01:

            if any(abs(e) > 0.001 for e in self.last_trajectory_errors):
                errors = self.last_trajectory_errors.copy()
                print(f"运动微小，使用上一次轨迹errors: {[f'{e:.4f}' for e in errors]}")
            else:
                print(f"持续静止，使用当前微小errors: {[f'{e:.4f}' for e in errors]}")
        else:
            self.last_trajectory_errors = errors.copy()
            print(f"检测到运动，保存当前errors: {[f'{e:.4f}' for e in errors]}")

        self.current_errors = errors.copy()
        self.last_valid_errors = errors.copy()

        for i in range(3):
            if abs(errors[i]) < self.death:
                errors[i] = 0

        adjusted_errors = [error * self.speed_multiplier for error in errors]

        vnorth_current = -adjusted_errors[2] * 100
        veast_current = -adjusted_errors[0] * 100
        vdown_current = adjusted_errors[1] * 100

        await self.send_drone_command(vnorth_current, veast_current, vdown_current, self.current_yaw_speed)

        self.vnorth_last = vnorth_current
        self.veast_last = veast_current
        self.vdown_last = vdown_current

    def _draw_hand_visualization(self, frame, hand_result, landmarks):
        should_draw_trajectory = (self.recording_trajectory and
                                  self.drone_state['current_gesture'] == '1' and
                                  len(self.trajectory_3d) > 1)

        if should_draw_trajectory:
            for i in range(1, len(self.trajectory_3d)):
                prev_pt, curr_pt = self.trajectory_3d[i - 1], self.trajectory_3d[i]

                if not (self.is_valid_point(prev_pt) and self.is_valid_point(curr_pt)):
                    continue

                try:
                    px_prev = rs.rs2_project_point_to_pixel(self.gaze_estimation_thread.color_intrinsics, prev_pt)
                    px_curr = rs.rs2_project_point_to_pixel(self.gaze_estimation_thread.color_intrinsics, curr_pt)

                    if not (self.is_valid_point(px_prev) and self.is_valid_point(px_curr)):
                        continue

                    avg_depth = (prev_pt[2] + curr_pt[2]) / 2.0
                    norm = np.clip((avg_depth - self.MIN_DEPTH) / (self.MAX_DEPTH - self.MIN_DEPTH), 0.0, 1.0)
                    rgba = self.colormap(norm)
                    color = tuple(int(255 * c) for c in rgba[:3])[::-1]

                    cv2.line(frame, (int(px_prev[0]), int(px_prev[1])),
                             (int(px_curr[0]), int(px_curr[1])), color, 2)
                except Exception as e:
                    continue

        self.mp_draw.draw_landmarks(frame, hand_result.multi_hand_landmarks[0],
                                    mp.solutions.hands.HAND_CONNECTIONS)

        lm = landmarks[8]
        u = int(lm.x * self.gaze_estimation_thread.color_intrinsics.width)
        v = int(lm.y * self.gaze_estimation_thread.color_intrinsics.height)

        gesture_colors = {
            'ok': (0, 255, 0),  # 绿色
            '1': (255, 0, 0),  # 蓝色
            'thumb': (0, 255, 255),  # 黄色
            'fist': (255, 0, 255),  # 紫色
            'brake': (0, 0, 255),  # 红色
            'noges': (128, 128, 128)  # 灰色
        }
        color = gesture_colors.get(self.drone_state['current_gesture'], (255, 255, 255))
        cv2.circle(frame, (u, v), 8, color, -1)

        if self.drone_state['current_gesture'] == 'noges':
            trajectory_status = f" | NOGES Mode: Keeping errors {self.current_errors}"
        elif self.drone_state['current_gesture'] == '1':
            if self.recording_trajectory:
                trajectory_status = f" | DRAWING: {len(self.trajectory_3d)} points"
            else:
                trajectory_status = " | Ready to Draw (new trajectory)"
        elif self.control_mode in ['fist', 'thumb']:
            trajectory_status = f" | {self.control_mode.upper()} Mode: Speed Control"
        elif self.drone_state['trajectory_mode'] and len(self.trajectory_3d) > 0:
            trajectory_status = f" | Using Existing Trajectory: {len(self.trajectory_3d)} points"
        else:
            trajectory_status = " | No Trajectory"

        speed_info = f" | Speed: {self.speed_multiplier:.1f}x | Mode: {self.control_mode.upper()}"
        full_text = trajectory_status + speed_info
        cv2.putText(frame, full_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    async def run(self):

        gaze_thread = threading.Thread(target=self.gaze_estimation_thread.run, daemon=True)
        gaze_thread.start()

        time.sleep(3.0)

        try:
            while rclpy.ok() and not self.control_events['exit'].is_set():
                try:
                    rclpy.spin_once(self.ros_comm, timeout_sec=0.01)
                    self.calculate_image_fps()
                    try:
                        self.latest_gaze_data = self.gaze_data_queue.get_nowait()
                    except queue.Empty:
                        pass

                    if self.latest_gaze_data is None:
                        self.latest_gaze_data = self._create_default_gaze_data()

                    display_image, detected_objects, timestamp = self.ros_comm.get_latest_data()

                    if display_image is not None:
                        vis_image, current_gazed_object = self.draw_gaze_visualization(display_image, self.latest_gaze_data, detected_objects)
                        self.ros_comm.publish_visualization(vis_image, timestamp)

                        self._publish_gaze_data(current_gazed_object)

                    processed_frame = await self.process_hand_gesture_with_realsense()
                    if processed_frame is not None:
                        self._display_realsense_window(processed_frame, detected_objects)

                    self._handle_keyboard_input()

                except Exception as e:
                    print(f"主循环错误: {e}")
                    time.sleep(0.01)

        finally:
            self.cleanup()

    def _create_default_gaze_data(self):
        return GazeData(
            timestamp=time.time(), gaze_point=None, img_gaze_point=None, gazed_object=None,
            eye_pos_3d=None, motion_state="unknown", smoothed_pitch=0.0, smoothed_yaw=0.0,
            raw_pitch=0.0, raw_yaw=0.0, corrected_pitch=0.0, corrected_yaw=0.0,
            face_bbox=None, face_score=0.0, tracking_status="无目标", yaw_control_mode="gaze",
            current_yaw_speed=0.0, calibration_mode=False, realsense_connected=False
        )

    def _publish_gaze_data(self, gazed_object=None):
        gaze_data_dict = {
            'timestamp': self.latest_gaze_data.timestamp,
            'gaze_point': self.latest_gaze_data.gaze_point,
            'img_gaze_point': self.latest_gaze_data.img_gaze_point,
            'gazed_object': self.latest_gaze_data.gazed_object,
            'motion_state': self.latest_gaze_data.motion_state,
            'instant_mode': self.gaze_matcher.fast_mode,
            'response_frames': 1,
            'gaze_fps': self.gaze_estimation_thread.current_fps,
            'image_fps': self.current_image_fps,
            'calibrated': self.gaze_estimation_thread.calibrator.is_calibrated,
            'architecture': 'Unified Errors Control System',
            'drone_ready': self.drone_ready,
            'drone_taken_off': self.drone_state['taken_off'],
            'trajectory_mode': self.drone_state['trajectory_mode'],
            'current_gesture': self.drone_state['current_gesture'],
            'speed_multiplier': self.speed_multiplier,  # 修改：发布speed_multiplier而不是speed_factor
            'hand_recording': self.recording_trajectory,
            'tracking_status': self.latest_gaze_data.tracking_status,
            'yaw_control_mode': self.latest_gaze_data.yaw_control_mode,
            'eye_yaw_speed': self.latest_gaze_data.current_yaw_speed,
            'calibration_mode': self.latest_gaze_data.calibration_mode,
            'realsense_connected': self.latest_gaze_data.realsense_connected,
            'gesture_model_loaded': self.gesture_model is not None,
            'control_mode': self.control_mode,  # 新增：发布当前控制模式
            'current_errors': self.current_errors,  # 新增：发布当前errors
            'visual_objects': {
		    'gazed': gazed_object,  # 直接从draw_gaze_visualization获取的gazed_object
		    'locked': self.object_tracker.current_target if self.object_tracker.is_locked else None,
		    'candidate': getattr(self.object_tracker, 'new_target_candidate', None) if hasattr(self.object_tracker, 'new_target_candidate') else None,
		    'is_locked_active': self.object_tracker.is_locked
	        }
        }
        self.ros_comm.publish_gaze_data(gaze_data_dict)

    def _display_realsense_window(self, processed_frame, detected_objects):

        cv2.imshow('Unified Errors Control: Gaze + Hand + Drone', processed_frame)

    def _handle_keyboard_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.control_events['exit'].set()
        elif key == ord('r'):
            self.control_events['reset'].set()
            self.gaze_matcher.reset()
            self.object_tracker.reset()
        elif key == ord('f'):
            self.gaze_matcher.set_fast_mode(not self.gaze_matcher.fast_mode)
        elif key == ord('k'):
            if not self.gaze_estimation_thread.realsense_initialized:
                print("RealSense未初始化，无法进行标定")
                return
            self.control_events['calibration_mode'].set()
            self.object_tracker.reset()
        elif key == ord(' '):
            self.control_events['calibration_data'].set()
        elif key == ord('t'):
            self.object_tracker.reset()

    def cleanup(self):
        try:
            self.control_events['exit'].set()

            if self.drone and self.drone_ready:
                try:
                    asyncio.create_task(self.send_drone_command(0, 0, 0, 0))
                except:
                    pass

            if hasattr(self, 'gaze_estimation_thread'):
                self.gaze_estimation_thread.stop()

            if hasattr(self, 'mp_hands'):
                self.mp_hands.close()

            if hasattr(self, 'ros_comm') and self.ros_initialized:
                self.ros_comm.destroy_node()

            if self.ros_initialized and rclpy.ok():
                rclpy.shutdown()

            cv2.destroyAllWindows()
            print("程序已退出")

        except Exception as e:
            print(f"清理资源时出错: {e}")


# 主函数
async def main():
    dependencies = [
        ('rclpy', 'ROS2'),
        ('mavsdk', 'MAVSDK')
    ]

    for module, name in dependencies:
        try:
            __import__(module)
        except ImportError:
            return

    model_path = pathlib.Path.cwd() / 'models' / 'L2CSNet_gaze360.pkl'
    if not model_path.exists():
        return

    gesture_model_path = pathlib.Path(MODEL_FILE)
    if not gesture_model_path.exists():
        return

    try:
        drone, success = await init_drone()
        if not success:
            return
        controller = ParallelIntegratedGazeDroneController(drone)
        await controller.run()

    except Exception as e:
        print("请检查：1.RealSense D435连接; 2.依赖库安装; 3.ROS2环境; 4.增强版YOLO检测器; 5.无人机连接; 6.手势识别模型")
    finally:
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
