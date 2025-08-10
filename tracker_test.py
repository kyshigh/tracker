import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
from cv_bridge import CvBridge
import time


class SearchAndTrackRobot(Node):
    def __init__(self):
        super().__init__('search_and_track_robot')

        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        # HSV 색상 범위 (빨간색)
        self.lower_red1 = np.array([0, 100, 100])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])
        self.upper_red2 = np.array([180, 255, 255])

        # 제어 변수
        self.min_area = 800
        self.kp_angular = 0.008
        self.max_angular = 1.2

        # 🎯 상태 관리
        self.robot_state = "SEARCHING"  # "SEARCHING" 또는 "TRACKING"
        self.last_detection_time = None
        self.lost_threshold = 1.0  # 1초 동안 못 찾으면 탐색 모드

        # 탐색 모드 설정
        self.search_angular_speed = 0.6  # 탐색시 회전 속도
        self.search_direction = 1  # 1: 왼쪽, -1: 오른쪽

        # 추적 모드 설정
        self.tracking_linear_speed = 0.2
        self.approach_threshold = 8000  # 이 영역 이상이면 멈춤
        self.far_threshold = 2000  # 이 영역 미만이면 빠르게 접근

        self.format_logged = False

        self.get_logger().info('🎯 Search & Track Robot 시작!')
        self.get_logger().info('🔍 탐색 모드로 시작합니다.')

    def image_callback(self, msg):
        try:
            if not self.format_logged:
                self.get_logger().info(f'📷 카메라 인코딩: {msg.encoding}')
                self.format_logged = True

            # 이미지 변환
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if msg.encoding in ['yuyv', 'yuv422', 'yuv422_yuy2']:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
            elif msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            height, width = frame.shape[:2]
            center_screen = width // 2

            # HSV 변환 및 마스크 생성
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            # 노이즈 제거
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 컨투어 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 물체 검출 여부 확인
            target_found = False
            twist = Twist()

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)

                if area > self.min_area:
                    target_found = True
                    self.last_detection_time = time.time()

                    # 중심점 계산
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        # 🎯 추적 모드로 전환
                        if self.robot_state == "SEARCHING":
                            self.robot_state = "TRACKING"
                            self.get_logger().info('🎯 목표 발견! 추적 모드로 전환')

                        # 추적 제어
                        twist = self.track_target(cx, cy, area, center_screen, width)

            # 상태 업데이트
            self.update_robot_state(target_found)

            # 물체가 없거나 너무 작으면 탐색 모드
            if not target_found:
                if self.robot_state == "SEARCHING":
                    twist = self.search_target()
                else:
                    # 추적 중이었는데 놓쳤을 때 - 잠시 정지 후 탐색
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

            # 제어 명령 발행
            self.cmd_pub.publish(twist)

        except Exception as e:
            self.get_logger().error(f'❌ 이미지 처리 오류: {e}')

    # track_target 함수의 거리별 속도 제어 부분을 이렇게 바꿀 수도 있습니다:

    def track_target(self, cx, cy, area, center_screen, width):
        """목표 추적 제어 - 부드러운 속도 제어"""
        twist = Twist()

        # 좌우 오차 계산
        error_x = cx - center_screen
        normalized_error = error_x / (width / 2)

        # 회전 제어
        twist.angular.z = -normalized_error * self.kp_angular
        twist.angular.z = max(-self.max_angular, min(self.max_angular, twist.angular.z))

        # 🐌 더 부드러운 거리별 속도 제어
        if area < 1500:
            # 매우 멀면
            twist.linear.x = 0.12
            distance_status = "VERY FAR - 아주 천천히"
        elif area < self.far_threshold:  # 2000
            # 멀면
            twist.linear.x = 0.10
            distance_status = "FAR - 천천히 접근"
        elif area < 5000:
            # 조금 멀면
            twist.linear.x = 0.08
            distance_status = "MEDIUM - 조금씩 접근"
        elif area > self.approach_threshold:  # 8000
            # 너무 가까우면
            twist.linear.x = -0.03
            distance_status = "CLOSE - 살짝 후진"
        else:
            # 적당한 거리
            twist.linear.x = 0.05
            distance_status = "PERFECT - 완벽한 거리"

        # 방향 상태
        if cx < center_screen - 50:
            direction_status = "LEFT ⬅️"
        elif cx > center_screen + 50:
            direction_status = "RIGHT ➡️"
        else:
            direction_status = "CENTER 🎯"

        self.get_logger().info(
            f'🎯 TRACKING | {direction_status} | {distance_status} | '
            f'위치: {cx:3d}/{width:3d} | 영역: {int(area):4d} | '
            f'속도: {twist.linear.x:4.2f} | 회전: {twist.angular.z:5.2f}'
        )

        return twist

    def search_target(self):
        """목표 탐색 제어"""
        twist = Twist()
        twist.linear.x = 0.0  # 제자리에서 회전
        twist.angular.z = self.search_angular_speed * self.search_direction

        self.get_logger().info(f'🔍 SEARCHING | 회전 탐색 중... (속도: {twist.angular.z:4.2f})')

        return twist

    def update_robot_state(self, target_found):
        """로봇 상태 업데이트"""
        current_time = time.time()

        if not target_found and self.robot_state == "TRACKING":
            # 추적 중인데 목표를 놓쳤을 때
            if self.last_detection_time and (current_time - self.last_detection_time) > self.lost_threshold:
                self.robot_state = "SEARCHING"
                self.get_logger().warn('❌ 목표 상실! 탐색 모드로 전환')

                # 탐색 방향 변경 (마지막으로 본 위치 반대 방향으로)
                self.search_direction *= -1

    def destroy_node(self):
        """종료 시 정지"""
        self.get_logger().info('🛑 로봇 정지 중...')
        twist = Twist()  # 모든 속도를 0으로
        self.cmd_pub.publish(twist)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    robot = SearchAndTrackRobot()

    print("\n🤖 Search & Track Robot")
    print("📋 동작 모드:")
    print("  🔍 SEARCHING: 빨간 물체를 찾기 위해 제자리 회전")
    print("  🎯 TRACKING: 빨간 물체를 발견하면 추적하며 이동")
    print("  ❌ 추적 중 물체를 1초 이상 놓치면 다시 탐색 모드")
    print("\n🚀 시작합니다!")

    try:
        rclpy.spin(robot)
    except KeyboardInterrupt:
        robot.get_logger().info('🔴 사용자가 종료했습니다.')
    finally:
        # 안전한 종료
        twist = Twist()
        robot.cmd_pub.publish(twist)
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()