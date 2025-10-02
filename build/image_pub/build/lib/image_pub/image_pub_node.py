#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import glob, os
import numpy as np
import cv2

class ImagePub(Node):
    def __init__(self):
        super().__init__('image_pub')

        # 공통
        self.declare_parameter('topic', '/image_raw')
        self.declare_parameter('fps', 20.0)

        # 입력 모드
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('use_video', False)
        self.declare_parameter('video_path', '')
        self.declare_parameter('use_dir', False)
        self.declare_parameter('dir_path', '')
        self.declare_parameter('glob_pattern', '*.png')   # 예: '**/*.png' 또는 'RB2*/.../*.png'
        self.declare_parameter('loop_dir', True)
        self.declare_parameter('glob_recursive', True)    # 재귀 글롭 사용 여부

        topic = self.get_parameter('topic').value
        self.pub = self.create_publisher(Image, topic, 10)
        self.bridge = CvBridge()

        self.mode = 'camera'
        fps = float(self.get_parameter('fps').value)

        use_video = bool(self.get_parameter('use_video').value)
        use_dir   = bool(self.get_parameter('use_dir').value)

        if use_dir:
            # 디렉터리 모드
            self.mode = 'dir'
            dir_path = os.path.abspath(str(self.get_parameter('dir_path').value))
            pattern = str(self.get_parameter('glob_pattern').value)
            recursive = bool(self.get_parameter('glob_recursive').value)
            self.loop_dir = bool(self.get_parameter('loop_dir').value)

            # 재귀 글롭: '**' 패턴 또는 하위 경로 와일드카드 매칭
            search_pattern = os.path.join(dir_path, pattern)
            self.files = sorted(glob.glob(search_pattern, recursive=recursive))

            if not self.files:
                self.get_logger().error(f'No files match: {search_pattern}')
                raise FileNotFoundError(f'No files match: {search_pattern}')

            self.idx = 0
            self.timer = self.create_timer(1.0 / max(fps, 1.0), self.tick_dir)
            self.get_logger().info(
                f'DIR mode: {len(self.files)} frames\n'
                f'  dir_path     = {dir_path}\n'
                f'  glob_pattern = {pattern}\n'
                f'  recursive    = {recursive}\n'
                f'  -> publishing to {topic} @ {fps} FPS'
            )

        elif use_video:
            # 동영상 모드
            self.mode = 'video'
            video_path = str(self.get_parameter('video_path').value)
            if not os.path.exists(video_path):
                raise FileNotFoundError(video_path)
            self.cap = cv2.VideoCapture(video_path)
            src_fps = self.cap.get(cv2.CAP_PROP_FPS) or fps
            self.timer = self.create_timer(1.0 / max(src_fps, 1.0), self.tick_video)
            self.get_logger().info(f'VIDEO mode: {video_path} -> {topic} @ ~{src_fps:.1f} FPS')
        else:
            # 카메라 모드
            cam_idx = int(self.get_parameter('camera_index').value)
            self.cap = cv2.VideoCapture(cam_idx)
            if not self.cap.isOpened():
                self.get_logger().error(f'Camera {cam_idx} open failed')
            self.timer = self.create_timer(1.0 / max(fps, 1.0), self.tick_cam)
            self.get_logger().info(f'CAM mode: /dev/video{cam_idx} -> {topic} @ {fps} FPS')

    def publish_frame(self, frame):
        # frame의 채널/자료형에 따라 올바른 encoding 선택
        if frame.ndim == 2:
            # 단일 채널
            if frame.dtype == np.uint16:
                enc = 'mono16'
            else:
                # 대부분 uint8
                if frame.dtype != np.uint8:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                enc = 'mono8'
        else:
            # 3채널 (BGR 컬러맵 등)
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            enc = 'bgr8'
        msg = self.bridge.cv2_to_imgmsg(frame, encoding=enc)
        self.pub.publish(msg)

    def tick_cam(self):
        if not self.cap.isOpened():
            return
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.publish_frame(frame)

    def tick_video(self):
        if not self.cap.isOpened():
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().info('Video ended; looping to start')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        self.publish_frame(frame)

    def tick_dir(self):
        if not self.files:
            return
        path = self.files[self.idx]

        # 1) 원본 그대로 읽기 (16-bit 깊이도 OK)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if img is None:
            self.get_logger().warn(f'Failed to read: {path}')
        else:
            # 2) 단일채널(깊이) 처리: 16-bit → 정규화 → 8-bit → 컬러맵 → BGR
            if len(img.shape) == 2:
                # 깊이 min/max 정규화 (보이는 맛)
                norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                d8 = norm.astype('uint8')
                # Turbo가 없으면 JET 써도 됨
                color = cv2.applyColorMap(d8, getattr(cv2, 'COLORMAP_TURBO', cv2.COLORMAP_JET))
                frame = color  # BGR 3채널
            else:
                # 이미 컬러/3채널인 경우 그대로
                frame = img

            self.publish_frame(frame)

        self.idx += 1
        if self.idx >= len(self.files):
            if self.loop_dir:
                self.idx = 0
            else:
                self.get_logger().info('Directory playback finished')
                self.destroy_timer(self.timer)

def main(args=None):
    rclpy.init(args=args)
    node = ImagePub()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()