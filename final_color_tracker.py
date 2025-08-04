#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
from cv_bridge import CvBridge

class FinalColorTracker(Node):
    def __init__(self):
        super().__init__('final_color_tracker')
        
        # 구독자 및 발행자 설정
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()  # cv_bridge 올바른 사용
        
        # HSV 색상 범위 설정 (빨간색)
        self.lower_red1 = np.array([0, 50, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        # PID 제어 변수
        self.min_area = 800
        self.kp_angular = 0.003
        self.format_logged = False
        
        self.get_logger().info('🔴 Final Color Tracker 시작!')
    
    def image_callback(self, msg):
        try:
            # 원본 인코딩 확인 및 변환
            if not self.format_logged:
                self.get_logger().info(f'📷 원본 인코딩: {msg.encoding}')
                self.format_logged = True
            
            # 자동 인코딩 변환 (passthrough 사용)
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            # YUYV나 다른 포맷이면 BGR로 변환
            if msg.encoding in ['yuyv', 'yuv422', 'yuv422_yuy2']:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
            elif msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                # 이미 BGR 포맷이므로 변환 불필요
                pass
            else:
                self.get_logger().warn(f'⚠️ 지원되지 않는 인코딩: {msg.encoding}')
                return
            
            height, width = frame.shape[:2]
            
            # HSV 변환
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 빨간색 마스크 생성 (두 개의 범위)
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = mask1 + mask2  # 노이즈 제거
            
            # 모폴로지 연산으로 노이즈 제거
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            twist = Twist()
            
            if contours:
                # 가장 큰 컨투어 찾기
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > self.min_area:
                    # 중심점 계산
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 중심에서 벗어난 정도 계산 (PID 제어)
                        center_x = cx
                        center_y = cy
                        
                        # 각속도 제어 (좌우 회전)
                        error_x = (width // 2) - center_x
                        twist.linear.x = 0.1  # 전진 속도
                        twist.angular.z = error_x * self.kp_angular
                        
                        # 각속도 제한
                        if area < 3000:
                            twist.linear.x = 0.15
                        elif area > 12000:
                            twist.linear.x = -0.1
                        else:
                            twist.linear.x = 0.0
                        
                        # 각속도 범위 제한
                        twist.angular.z = max(-0.5, min(0.5, twist.angular.z))
                        
                        self.get_logger().info(f'🎯 추적 중: 좌표=({center_x}, {center_y}), 영역={int(area)}')
                else:
                    self.get_logger().warn(f'🔍 물체 크기 너무 작음: 영역={int(area)} < {self.min_area}')
            
            # 제어 명령 발행
            self.cmd_pub.publish(twist)
            
        except Exception as e:
            self.get_logger().error(f'❌ 이미지 처리 오류: {e}')
    
    def destroy_node(self):
        self.get_logger().info('🛑 Final Color Tracker 종료')
        twist = Twist()  # 정지 명령
        self.cmd_pub.publish(twist)
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    tracker = FinalColorTracker()
    
    try:
        rclpy.spin(tracker)
    except KeyboardInterrupt:
        tracker.get_logger().info('🔴 Color Tracker 종료!')
    finally:
        # 정지 명령
        twist = Twist()
        tracker.cmd_pub.publish(twist)
        tracker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
