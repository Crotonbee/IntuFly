#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
import json


class GazeboYOLODetector(Node):
    def __init__(self):
        super().__init__('gazebo_yolo_detector')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f"Using device: {self.device}")
        if self.device == 'cuda':
            self.get_logger().info(f"GPU model: {torch.cuda.get_device_name(0)}")
            self.get_logger().info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        self.bridge = CvBridge()

        self.get_logger().info("Loading YOLOv8X model...")
        self.model = YOLO('best.pt')
        if self.device == 'cuda':
            self.model.to(self.device)

        self.image_subscriber = self.create_subscription(
            Image,
            '/world/racing_world/model/x500_depth_3d_lidar_0/link/camera_link/sensor/IMX214/image',
            self.image_callback,
            10
        )

        self.image_publisher = self.create_publisher(
            Image,
            '/yolo_detection_result',
            10
        )

        self.detection_data_publisher = self.create_publisher(
            String,
            '/yolo_detection_data',
            10
        )

        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        self.total_detections = 0
        self.processing_times = []

        self.get_logger().info("Enhanced YOLO Detector initialized and ready!")
        self.get_logger().info("Subscribed to: /world/racing_world/model/x500_depth_3d_lidar_0/link/camera_link/sensor/IMX214/image")
        self.get_logger().info("Publishing image to: /yolo_detection_result")
        self.get_logger().info("Publishing detection data to: /yolo_detection_data")

    def calculate_fps(self):
        self.fps_counter += 1
        if self.fps_counter % 10 == 0:
            current_time = time.time()
            self.current_fps = 10 / (current_time - self.fps_start_time)
            self.fps_start_time = current_time

    def image_callback(self, msg):
        try:
            start_time = time.time()

            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            height, width = cv_image.shape[:2]

            results = self.model(cv_image, verbose=False, conf=0.3, device=self.device)

            detected_objects = 0
            detection_data = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    detected_objects = len(boxes)
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.model.names[class_id]

                        detection_data.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(conf),
                            'class_name': class_name,
                            'class_id': class_id
                        })

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        thickness = max(2, width // 400)
                        cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), thickness)

                        radius = max(3, width // 200)
                        cv2.circle(cv_image, (center_x, center_y), radius, (0, 0, 255), -1)

            detection_msg = String()
            detection_msg.data = json.dumps({
                'timestamp': time.time(),
                'image_width': width,
                'image_height': height,
                'detections': detection_data
            })
            self.detection_data_publisher.publish(detection_msg)

            self.calculate_fps()
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)

            if len(self.processing_times) > 100:
                self.processing_times.pop(0)

            self.total_detections += detected_objects

            info_text = [
                f"FPS: {self.current_fps:.1f}",
            ]

            font_scale = min(0.8, max(0.4, width / 1400))
            thickness_text = 2
            line_height = max(20, int(35 * font_scale))
            
            # 计算文本区域的边距
            margin_x = max(10, int(width * 0.01))
            margin_y = max(25, int(height * 0.03))

            for i, text in enumerate(info_text):
                y_pos = margin_y + i * line_height

                if y_pos > height - 10:
                    break

                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_text)

                if margin_x + text_width > width - 10:
                    adjusted_font_scale = font_scale * (width - margin_x - 10) / text_width
                    adjusted_font_scale = max(0.3, adjusted_font_scale)
                else:
                    adjusted_font_scale = font_scale

                cv2.putText(cv_image, text, (margin_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, adjusted_font_scale, (0, 0, 0), thickness_text + 2)
                cv2.putText(cv_image, text, (margin_x, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, adjusted_font_scale, (255, 255, 255), thickness_text)
            output_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            output_msg.header = msg.header
            self.image_publisher.publish(output_msg)

            if self.fps_counter % 30 == 0:
                avg_process_time = np.mean(self.processing_times) * 1000
                self.get_logger().info(
                    f"FPS: {self.current_fps:.1f}, Objects: {detected_objects}, "
                    f"Avg Process: {avg_process_time:.1f}ms, Total Detected: {self.total_detections}"
                )

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")


def main(args=None):
    print("Initializing Enhanced YOLOv8X + Gazebo Camera Detection System...")
    
    rclpy.init(args=args)
    
    try:
        detector = GazeboYOLODetector()
        
        rclpy.spin(detector)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        if rclpy.ok():
            detector.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
