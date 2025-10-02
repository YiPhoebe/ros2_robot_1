#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class OverlayViz(Node):
    def __init__(self):
        super().__init__('overlay_viz_node')

        # 파라미터 선언
        self.declare_parameter('overlay_conf_min', 0.25)
        self.declare_parameter('input_topic', '/image_raw')
        self.declare_parameter('output_topic', '/image_overlay')

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.bridge = CvBridge()

        # 구독자
        self.image_sub = self.create_subscription(
            Image, input_topic, self.image_callback, 10)

        # 퍼블리셔
        self.image_pub = self.create_publisher(Image, output_topic, 10)

        self.get_logger().info(f"Subscribed to {input_topic}, publishing to {output_topic}")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # TODO: YOLO 결과 박스 → cv2.rectangle() 추가
        cv2.putText(frame, "OverlayViz Active", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        overlay_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.image_pub.publish(overlay_msg)
        # 로그는 너무 많이 안 찍도록
        self.get_logger().debug("Published overlay image")

def main(args=None):
    rclpy.init(args=args)
    node = OverlayViz()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()