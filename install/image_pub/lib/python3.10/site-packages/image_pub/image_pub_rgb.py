#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os

class ImagePubRGB(Node):
    def __init__(self):
        super().__init__('image_pub_rgb')

        self.declare_parameter('video_path', '')
        self.declare_parameter('topic', '/image_rgb')

        video_path = str(self.get_parameter('video_path').value)
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f'Failed to open video: {video_path}')

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        self.pub = self.create_publisher(Image, self.get_parameter('topic').value, 10)
        self.bridge = CvBridge()

        self.timer = self.create_timer(1.0 / fps, self.tick)
        self.get_logger().info(f'Publishing {video_path} to {self.get_parameter("topic").value} @ {fps:.1f} FPS')

    def tick(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePubRGB()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()