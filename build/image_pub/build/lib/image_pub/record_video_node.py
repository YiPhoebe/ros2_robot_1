#!/usr/bin/env python3
import os, cv2, numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')
        self.declare_parameter('topic', '/image_overlay')   # 저장할 토픽
        self.declare_parameter('out_path', '/workspace/output/out.mp4')
        self.declare_parameter('fps', 10.0)
        self.declare_parameter('fourcc', 'mp4v')            # 'mp4v','XVID','MJPG' 등
        topic   = self.get_parameter('topic').value
        self.out_path = self.get_parameter('out_path').value
        base, ext = os.path.splitext(self.out_path)
        self.out_path = f"{base}_timelapse{ext}"
        self.fps  = float(self.get_parameter('fps').value)
        fourcc = cv2.VideoWriter_fourcc(*str(self.get_parameter('fourcc').value))

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        self.bridge = CvBridge()
        self.writer = None
        self.fourcc = fourcc

        self.sub = self.create_subscription(Image, topic, self.cb, 10)
        self.get_logger().info(f"Recording '{topic}' -> {self.out_path} @ {self.fps} FPS")

    def _ensure_writer(self, h, w):
        if self.writer is None:
            self.writer = cv2.VideoWriter(self.out_path, self.fourcc, self.fps, (w, h), True)
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter: {self.out_path}")

    def cb(self, msg: Image):
        # 어떤 encoding이 오든 최종적으로 8-bit BGR로 저장
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

        if img.ndim == 2:
            # mono8/mono16 둘 다 지원 → 보기 좋게 컬러맵해서 BGR 저장
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img = cv2.applyColorMap(img, getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET))
        else:
            # 컬러가 16비트 등인 경우도 8비트로 변환
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

        h, w = img.shape[:2]
        self._ensure_writer(h, w)
        self.writer.write(img)

    def destroy_node(self):
        if self.writer is not None:
            self.writer.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorder()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()