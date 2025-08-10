import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
import cv2
import numpy as np
from cv_bridge import CvBridge


class FixedColorTracker(Node):
    def __init__(self):
        super().__init__('fixed_color_tracker')

        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()

        # 개선된 HSV 색상 범위 (더 넓은 범위)
        self.lower_red1 = np.array([0, 100, 100])  # 더 진한 빨간색만
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([160, 100, 100])  # 범위 확장
        self.upper_red2 = np.array([180, 255, 255])

        # 최적화된 제어 변수
        self.min_area = 500  # 더 작은 물체도 추적
        self.kp_angular = 0.008  # 더 민감한 반응
        self.max_angular = 1.0  # 더 빠른 회전
        self.linear_speed = 0.2  # 더 빠른 전진

        self.format_logged = False

        self.get_logger().info('🚀 Fixed Color Tracker 시작!')

    def image_callback(self, msg):
        try:
            if not self.format_logged:
                self.get_logger().info(f'📷 인코딩: {msg.encoding}')
                self.format_logged = True

            # 이미지 변환
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

            if msg.encoding in ['yuyv', 'yuv422', 'yuv422_yuy2']:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUYV)
            elif msg.encoding == 'rgb8':
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            height, width = frame.shape[:2]
            center_screen = width // 2

            # HSV 변환
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 빨간색 마스크 (두 범위 합치기)
            mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            # 강화된 노이즈 제거
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 가우시안 블러로 부드럽게
            mask = cv2.GaussianBlur(mask, (5, 5), 0)

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

                        # 🎯 핵심: 오차 계산 (방향 확인!)
                        error_x = cx - center_screen  # 물체위치 - 화면중앙

                        # 정규화된 오차 (-1 ~ +1)
                        normalized_error = error_x / (width / 2)

                        # 제어 명령 생성
                        twist.linear.x = self.linear_speed  # 항상 전진
                        twist.angular.z = -normalized_error * self.kp_angular  # 음수로 방향 조정

                        # 각속도 제한
                        twist.angular.z = max(-self.max_angular,
                                              min(self.max_angular, twist.angular.z))

                        # 상태 표시
                        if cx < center_screen - 30:
                            direction = "LEFT ⬅️"
                            action = "왼쪽 회전"
                        elif cx > center_screen + 30:
                            direction = "RIGHT ➡️"
                            action = "오른쪽 회전"
                        else:
                            direction = "CENTER 🎯"
                            action = "직진"

                        self.get_logger().info(
                            f'{direction} | 물체: {cx:3d}/{width:3d} | '
                            f'오차: {error_x:4.0f} | 회전: {twist.angular.z:5.2f} | '
                            f'동작: {action}'
                        )

                else:
                    self.get_logger().warn(f'❌ 물체 너무 작음: {int(area)} < {self.min_area}')
                    # 제자리에서 천천히 회전하며 탐색
                    twist.linear.x = 0.0
                    twist.angular.z = 0.3

            else:
                self.get_logger().warn('🔍 빨간 물체 미발견 - 탐색 중')
                # 물체 없으면 제자리 회전
                twist.linear.x = 0.0
                twist.angular.z = 0.5

            # 제어 명령 발행
            self.cmd_pub.publish(twist)

        except Exception as e:
            self.get_logger().error(f'❌ 오류: {e}')

    def destroy_node(self):
        # 정지 명령
        twist = Twist()
        self.cmd_pub.publish(twist)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    tracker = FixedColorTracker()

    try:
        rclpy.spin(tracker)
    except KeyboardInterrupt:
        tracker.get_logger().info('🛑 종료!')
    finally:
        twist = Twist()
        tracker.cmd_pub.publish(twist)
        tracker.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()