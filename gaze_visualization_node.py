#!/usr/bin/env python3
"""
增强版专用可视化节点 - 合并YOLO检测结果和注视点数据，包含物体高亮和锁定可视化
订阅: /yolo_detection_result (图像), /yolo_detection_data (检测数据), /gaze_data (注视数据)
发布: /combined_gaze_visualization (合并可视化)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import json
import time
import threading
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


class GazeVisualizationNode(Node):
    def __init__(self):
        super().__init__('gaze_visualization_node')
        
        # 初始化CV Bridge
        self.bridge = CvBridge()
        
        # QoS配置 - 优化性能
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 数据锁和存储
        self.data_lock = threading.Lock()
        self.latest_image = None
        self.latest_gaze_data = None
        self.detected_objects = []
        self.image_timestamp = None
        self.gaze_timestamp = None
        
        # 屏幕和图像坐标转换参数 (从allhand.py中获取)
        self.screen_config = {
            'width': 2560, 
            'height': 1440
        }
        
        # 订阅者
        self.image_subscriber = self.create_subscription(
            Image, 
            '/yolo_detection_result', 
            self.image_callback, 
            qos_profile
        )
        
        self.gaze_subscriber = self.create_subscription(
            String,
            '/gaze_data',
            self.gaze_callback,
            qos_profile
        )
        
        # 新增：订阅检测数据
        self.detection_subscriber = self.create_subscription(
            String,
            '/yolo_detection_data',
            self.detection_callback,
            qos_profile
        )
        
        # 发布者
        self.combined_publisher = self.create_publisher(
            Image,
            '/combined_gaze_visualization',
            qos_profile
        )
        
        # 性能统计
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        self.get_logger().info("Enhanced Gaze Visualization Node initialized!")
        self.get_logger().info("Subscribed to: /yolo_detection_result, /yolo_detection_data, /gaze_data")
        self.get_logger().info("Publishing to: /combined_gaze_visualization")

    def image_callback(self, msg):
        """接收YOLO处理后的图像"""
        try:
            with self.data_lock:
                self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                self.image_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                
            # 有图像数据时尝试生成可视化
            self.generate_visualization()
            
        except Exception as e:
            self.get_logger().error(f"Image callback error: {e}")

    def gaze_callback(self, msg):
        """接收注视数据"""
        try:
            with self.data_lock:
                self.latest_gaze_data = json.loads(msg.data)
                self.gaze_timestamp = self.latest_gaze_data.get('timestamp', time.time())
                
        except Exception as e:
            self.get_logger().error(f"Gaze callback error: {e}")

    def detection_callback(self, msg):
        """接收检测数据"""
        try:
            with self.data_lock:
                data = json.loads(msg.data)
                self.detected_objects = data.get('detections', [])
        except Exception as e:
            self.get_logger().error(f"Detection callback error: {e}")

    def convert_gaze_to_image_coordinates(self, gaze_point, image_width, image_height):
        """将屏幕坐标转换为图像坐标"""
        if not gaze_point:
            return None
            
        try:
            screen_x, screen_y = gaze_point
            # 坐标转换逻辑 (从allhand.py复制)
            img_x = screen_x * image_width / self.screen_config['width']
            img_y = screen_y * image_height / self.screen_config['height']
            img_x = max(0, min(image_width - 1, img_x))
            img_y = max(0, min(image_height - 1, img_y))
            return (img_x, img_y)
        except Exception as e:
            self.get_logger().error(f"Coordinate conversion error: {e}")
            return None

    def draw_gaze_point(self, image, gaze_point, calibration_mode=False):
        """绘制注视点"""
        if not gaze_point:
            return
            
        height, width = image.shape[:2]
        img_gaze_point = self.convert_gaze_to_image_coordinates(gaze_point, width, height)
        
        if img_gaze_point:
            x, y = int(img_gaze_point[0]), int(img_gaze_point[1])
            x, y = max(0, min(width - 1, x)), max(0, min(height - 1, y))
            
            # 根据校准模式设置颜色
            color = (0, 255, 0) if not calibration_mode else (0, 255, 255)
            
            # 绘制注视点
            cv2.circle(image, (x, y), 15, color, 3)
            cv2.circle(image, (x, y), 2, color, -1)
            
            # 显示坐标信息
            if gaze_point and not calibration_mode:
                coord_text = f"Gaze:({gaze_point[0]:.0f},{gaze_point[1]:.0f})"
                cv2.putText(image, coord_text, (x + 25, y - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_object_label(self, image, label, x, y, color):
        """绘制物体标签 (从allhand.py复制)"""
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(image, (x, y - 35), (x + label_size[0], y), color, -1)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def is_same_object_for_display(self, obj1, obj2):
        """判断是否为同一物体 (从allhand.py复制)"""
        if not obj1 or not obj2 or obj1['class_name'] != obj2['class_name']:
            return False
        center1 = ((obj1['bbox'][0] + obj1['bbox'][2]) / 2, (obj1['bbox'][1] + obj1['bbox'][3]) / 2)
        center2 = ((obj2['bbox'][0] + obj2['bbox'][2]) / 2, (obj2['bbox'][1] + obj2['bbox'][3]) / 2)
        distance = math.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
        return distance < 100

    def find_locked_object(self, detected_objects, tracking_status):
        """根据tracking_status找到锁定的物体"""
        if not tracking_status or "锁定" not in tracking_status:
            return None
            
        # 解析tracking_status，提取锁定的物体类别名
        try:
            if "🎯锁定:" in tracking_status:
                # 格式: "🎯锁定: person" 或 "🎯锁定: person | 候选: cup (30%)"
                locked_part = tracking_status.split("🎯锁定:")[1].split("|")[0].strip()
                locked_class_name = locked_part.strip()
                
                # 在detected_objects中找到对应的物体
                for obj in detected_objects:
                    if obj['class_name'] == locked_class_name:
                        return obj
        except:
            pass
            
        return None

    def get_candidate_object(self, tracking_status, detected_objects):
        """获取候选目标物体"""
        if not tracking_status or "候选:" not in tracking_status:
            return None
            
        try:
            if "候选:" in tracking_status:
                # 格式: "🎯锁定: person | 候选: cup (30%)"
                candidate_part = tracking_status.split("候选:")[1].strip()
                candidate_class_name = candidate_part.split("(")[0].strip()
                
                for obj in detected_objects:
                    if obj['class_name'] == candidate_class_name:
                        return obj
        except:
            pass
            
        return None

    # def highlight_object(self, image, gazed_object, tracking_status, detected_objects):
    #     """高亮物体 (从allhand.py复制并修改)"""
    #     if not gazed_object:
    #         return
    #
    #     x1, y1, x2, y2 = [int(coord) for coord in gazed_object['bbox']]
    #
    #     # 判断是否为锁定状态
    #     is_locked = tracking_status and "🎯锁定:" in tracking_status
    #
    #     if is_locked:
    #         # 检查当前注视物体是否为锁定目标
    #         locked_obj = self.find_locked_object(detected_objects, tracking_status)
    #         if locked_obj and self.is_same_object_for_display(locked_obj, gazed_object):
    #             # 锁定目标 - 红色边框，闪烁黄色外框
    #             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    #             if int(time.time() * 4) % 2:  # 闪烁效果
    #                 cv2.rectangle(image, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (255, 255, 0), 3)
    #             label = f"🎯 LOCKED: {gazed_object['class_name']} ({gazed_object['confidence']:.2f})"
    #             self.draw_object_label(image, label, x1, y1, (0, 0, 255))
    #             return
    #
    #     # 检查是否为候选目标
    #     candidate_obj = self.get_candidate_object(tracking_status, detected_objects)
    #     if candidate_obj and self.is_same_object_for_display(candidate_obj, gazed_object):
    #         # 候选目标 - 橙色边框
    #         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 3)
    #         label = f"🔄 CANDIDATE: {gazed_object['class_name']} ({gazed_object['confidence']:.2f})"
    #         self.draw_object_label(image, label, x1, y1, (0, 165, 255))
    #         return
    #
    #     # 普通注视目标 - 黄色边框
    #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
    #     label = f"GAZING: {gazed_object['class_name']} ({gazed_object['confidence']:.2f})"
    #     self.draw_object_label(image, label, x1, y1, (0, 255, 255))

    def highlight_object_with_visual_data(self, image, gazed_object, locked_object, candidate_object, is_locked_active):
        """使用visual_objects数据直接高亮物体"""
        x1, y1, x2, y2 = [int(coord) for coord in gazed_object['bbox']]

        # 判断当前注视对象的类型
        if is_locked_active and locked_object and self.is_same_object_for_display(gazed_object, locked_object):
            # 锁定目标
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
            if int(time.time() * 4) % 2:
                cv2.rectangle(image, (x1 - 5, y1 - 5), (x2 + 5, y2 + 5), (255, 255, 0), 3)
            label = f"🎯 LOCKED: {gazed_object['class_name']} ({gazed_object['confidence']:.2f})"
            self.draw_object_label(image, label, x1, y1, (0, 0, 255))
        elif candidate_object and self.is_same_object_for_display(gazed_object, candidate_object):
            # 候选目标
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 165, 255), 3)
            label = f"🔄 CANDIDATE: {gazed_object['class_name']} ({gazed_object['confidence']:.2f})"
            self.draw_object_label(image, label, x1, y1, (0, 165, 255))
        else:
            # 普通注视目标
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
            label = f"GAZING: {gazed_object['class_name']} ({gazed_object['confidence']:.2f})"
            self.draw_object_label(image, label, x1, y1, (0, 255, 255))

    def highlight_locked_object_without_gaze_new(self, image, locked_object):
        """直接使用locked_object数据高亮"""
        x1, y1, x2, y2 = [int(coord) for coord in locked_object['bbox']]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 150), 4)
        label = f"🎯 LOCKED (No Gaze): {locked_object['class_name']} ({locked_object['confidence']:.2f})"
        self.draw_object_label(image, label, x1, y1, (0, 0, 150))

    def highlight_locked_object_without_gaze(self, image, detected_objects, tracking_status):
        """高亮没有被注视但被锁定的目标"""
        locked_obj = self.find_locked_object(detected_objects, tracking_status)
        if not locked_obj:
            return
            
        x1, y1, x2, y2 = [int(coord) for coord in locked_obj['bbox']]
        # 锁定但未注视 - 深蓝色边框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 150), 4)
        label = f"🎯 LOCKED (No Gaze): {locked_obj['class_name']} ({locked_obj['confidence']:.2f})"
        self.draw_object_label(image, label, x1, y1, (0, 0, 150))

    def draw_gazed_objects(self, image, gaze_data, detected_objects):
        if gaze_data and gaze_data.get('calibration_mode'):
            return

        # 直接从visual_objects获取数据
        visual_objects = gaze_data.get('visual_objects', {}) if gaze_data else {}
        gazed_object = visual_objects.get('gazed')
        locked_object = visual_objects.get('locked')
        candidate_object = visual_objects.get('candidate')
        is_locked_active = visual_objects.get('is_locked_active', False)
        
        if gazed_object:
            self.highlight_object_with_visual_data(image, gazed_object, locked_object, candidate_object, is_locked_active)
        elif locked_object and is_locked_active:
            self.highlight_locked_object_without_gaze_new(image, locked_object)

    def draw_status_info(self, image, gaze_data):
        """绘制状态信息"""
        if not gaze_data:
            return
            
        height, width = image.shape[:2]
        
        # 基础状态信息
        status_items = [
            #f"Combined Visualization - FPS: {self.current_fps:.1f}",
            #f"Gaze Point: {'✓' if gaze_data.get('gaze_point') else '✗'}",
            #f"RealSense: {'✓' if gaze_data.get('realsense_connected') else '✗'}",
            #f"Calibrated: {'✓' if gaze_data.get('calibrated') else '✗'}",
            #f"Calibration Mode: {'ON' if gaze_data.get('calibration_mode') else 'OFF'}",
            #f"Motion: {gaze_data.get('motion_state', 'Unknown').upper()}",
            #f"Drone: {'Ready' if gaze_data.get('drone_ready') else 'Not Ready'}",
            #f"Gesture: {gaze_data.get('current_gesture', 'Unknown').upper()}",
            #f"Control Mode: {gaze_data.get('control_mode', 'Unknown').upper()}",
            #f"Speed: {gaze_data.get('speed_multiplier', 0):.1f}x",
            #f"Tracking: {gaze_data.get('tracking_status', 'No Target')}",
            #f"Yaw Mode: {gaze_data.get('yaw_control_mode', 'gaze').upper()}",
            #f"Eye Yaw Speed: {gaze_data.get('eye_yaw_speed', 0):.1f}°/s",
            #f"Objects: {len(self.detected_objects)} detected"
        ]
        
        # 绘制状态信息
        for i, status in enumerate(status_items):
            # 根据内容设置颜色
            color = (0, 255, 255)  # 默认青色
            if '✓' in status or 'Ready' in status or 'ON' in status:
                color = (0, 255, 0)  # 绿色
            elif '✗' in status or 'Not Ready' in status:
                color = (0, 0, 255)  # 红色
            elif 'Control Mode:' in status:
                # 根据控制模式设置不同颜色
                mode_colors = {
                    'TRAJECTORY': (0, 255, 0),
                    'THUMB': (0, 255, 255),
                    'FIST': (255, 0, 255),
                    'NOGES': (128, 128, 128),
                    'BRAKE': (0, 0, 255),
                    'IDLE': (255, 255, 255)
                }
                for mode, mode_color in mode_colors.items():
                    if mode in status:
                        color = mode_color
                        break
            elif 'Tracking:' in status:
                # 根据跟踪状态设置颜色
                if '🎯锁定' in status:
                    color = (0, 0, 255)  # 红色表示锁定
                elif '🔄' in status:
                    color = (0, 165, 255)  # 橙色表示跟踪
                elif 'No Target' in status:
                    color = (128, 128, 128)  # 灰色表示无目标
            
            cv2.putText(image, status, (10, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def draw_calibration_info(self, image, gaze_data):
        """绘制校准相关信息"""
        if not gaze_data.get('calibration_mode'):
            return
            
        height, width = image.shape[:2]
        
        # 校准模式警告
        warning_texts = [
            "CALIBRATION MODE ACTIVE",
            "Yaw control disabled during calibration",
            "Look at calibration points and press SPACE",
            "Press 'k' to exit calibration mode"
        ]
        
        # 绘制警告背景
        cv2.rectangle(image, (10, height - 120), (width - 10, height - 10), (0, 0, 0), -1)
        cv2.rectangle(image, (10, height - 120), (width - 10, height - 10), (0, 255, 255), 2)
        
        for i, text in enumerate(warning_texts):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            if i == 0:  # 第一行用红色高亮
                color = (0, 0, 255)
            cv2.putText(image, text, (20, height - 90 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def calculate_fps(self):
        """计算FPS"""
        self.frame_count += 1
        if self.frame_count % 15 == 0:
            current_time = time.time()
            elapsed_time = current_time - self.fps_start_time
            self.current_fps = 15 / elapsed_time
            self.fps_start_time = current_time

    def generate_visualization(self):
        """生成合并可视化"""
        with self.data_lock:
            if self.latest_image is None:
                return
                
            # 复制图像以避免修改原始数据
            vis_image = self.latest_image.copy()
            gaze_data = self.latest_gaze_data
            detected_objects = self.detected_objects.copy()
        
        try:
            # 绘制注视点
            if gaze_data and gaze_data.get('gaze_point'):
                self.draw_gaze_point(
                    vis_image, 
                    gaze_data['gaze_point'], 
                    gaze_data.get('calibration_mode', False)
                )
            
            # 绘制注视物体高亮和锁定可视化
            self.draw_gazed_objects(vis_image, gaze_data, detected_objects)
            
            # 绘制状态信息
            self.draw_status_info(vis_image, gaze_data)
            
            # 绘制校准信息
            if gaze_data:
                self.draw_calibration_info(vis_image, gaze_data)
            
            # 如果没有注视点，显示提示信息
            if not gaze_data or not gaze_data.get('gaze_point'):
                height, width = vis_image.shape[:2]
                status_text = "No gaze point detected"
                if gaze_data and not gaze_data.get('realsense_connected'):
                    status_text = "RealSense not connected"
                cv2.putText(vis_image, status_text, (10, height - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 计算FPS
            self.calculate_fps()
            
            # 发布合并后的图像
            output_msg = self.bridge.cv2_to_imgmsg(vis_image, encoding='bgr8')
            output_msg.header.stamp = self.get_clock().now().to_msg()
            self.combined_publisher.publish(output_msg)
            
            # 定期输出日志
            if self.frame_count % 60 == 0:
                gaze_status = "✓" if (gaze_data and gaze_data.get('gaze_point')) else "✗"
                locked_status = "🎯" if (gaze_data and "🎯锁定" in gaze_data.get('tracking_status', '')) else ""
                self.get_logger().info(
                    f"Visualization FPS: {self.current_fps:.1f}, "
                    f"Gaze: {gaze_status}, Objects: {len(detected_objects)}, "
                    f"Status: {locked_status}{gaze_data.get('tracking_status', 'No Target') if gaze_data else 'No Data'}"
                )
                
        except Exception as e:
            self.get_logger().error(f"Visualization generation error: {e}")


def main(args=None):
    print("Starting Enhanced Gaze Visualization Node...")
    
    rclpy.init(args=args)
    
    try:
        node = GazeVisualizationNode()
        
        print("Enhanced Gaze Visualization Node ready!")
        print("Subscribing to:")
        print("  - /yolo_detection_result (YOLO processed images)")
        print("  - /yolo_detection_data (YOLO detection data)")
        print("  - /gaze_data (gaze tracking data)")
        print("Publishing to:")
        print("  - /combined_gaze_visualization (combined visualization with object tracking)")
        print("Features:")
        print("  - Gaze point visualization")
        print("  - Object highlighting (gazing/candidate/locked)")
        print("  - Object tracking status display")
        print("  - Calibration mode support")
        print("Press Ctrl+C to stop")
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        print("Enhanced Gaze Visualization Node stopped")


if __name__ == "__main__":
    main()
