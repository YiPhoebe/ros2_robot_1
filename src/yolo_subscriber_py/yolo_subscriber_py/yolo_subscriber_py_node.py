#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import time

# 표준 메시지 권장: vision_msgs (없으면 package.xml에 추가)
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D

# YOLO: ultralytics (컨테이너에 pip install ultralytics 필요)
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

class YoloSubscriber(Node):
    def __init__(self):
        super().__init__('yolo_subscriber_py_node')

        # params - 코드 기본값은 안전하게, 실행 시엔 실험적으로
        # 실행할땐 런타임 override(덮어쓰기), 프로젝트마다 best threshold 다르니깐 유동적으로
        """
        ros2 run yolo_subscriber_py yolo_subscriber_py_node --ros-args \
          -p model_path:=/workspace/models/yolov8n.pt \
          -p device:=cuda:0 \
          -p conf:=0.3 \
          -p topic_in:=/camera/image_raw
        """
        self.declare_parameter('model_path', 'yolov8n.pt')   # 또는 yolov11n.pt 등
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('every_n', 1)                 # n 프레임마다 추론
        self.declare_parameter('device', 'cpu')              # 'cpu' or 'cuda:0'
        self.declare_parameter('topic_in', '/image_raw')
        self.declare_parameter('topic_out', '/yolo/bounding_boxes')  # Detection2DArray
        self.declare_parameter('classes', [])                # 특정 클래스만 (예: [0,1,2])
        self.declare_parameter('iou', 0.45)
        self.declare_parameter('half', False)

        self.bridge = CvBridge()
        self.frame_count = 0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        topic_in = self.get_parameter('topic_in').value
        self.sub = self.create_subscription(Image, topic_in, self.cb_image, qos)

        topic_out = self.get_parameter('topic_out').value
        self.pub = self.create_publisher(Detection2DArray, topic_out, 10)

        # YOLO 로드
        model_path = self.get_parameter('model_path').value
        if YOLO is None:
            raise RuntimeError("ultralytics 가 설치되지 않았습니다. `pip install ultralytics` 후 다시 실행하세요.")
        self.model = YOLO(model_path)

        device = str(self.get_parameter('device').value)
        try:
            self.model.to(device)
        except Exception:
            self.get_logger().warn(f"device 이동 실패({device}). CPU로 fallback 합니다.")
        
        # warmup
        imgsz = int(self.get_parameter('imgsz').value)
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        _ = self.model.predict(dummy, imgsz=imgsz, verbose=False)

        self.get_logger().info(
            f"YOLO ready: {model_path} | in:{topic_in} → out:{topic_out} | "
            f"conf={self.get_parameter('conf').value}, imgsz={imgsz}, every_n={self.get_parameter('every_n').value}"
        )

    def cb_image(self, msg: Image):
        self.frame_count += 1
        every_n = int(self.get_parameter('every_n').value)
        if every_n > 1 and (self.frame_count % every_n != 0):
            return

        # ROS Image → OpenCV
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"cv_bridge 변환 실패: {e}")
            return

        conf = float(self.get_parameter('conf').value)
        imgsz = int(self.get_parameter('imgsz').value)
        iou = float(self.get_parameter('iou').value)
        half = bool(self.get_parameter('half').value)
        classes = self.get_parameter('classes').value or None  # 빈 리스트면 None

        t0 = time.time()
        try:
            results = self.model.predict(
                cv_img,
                conf=conf,
                imgsz=imgsz,
                iou=iou,
                half=half,
                classes=classes,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f"YOLO 추론 실패: {e}")
            return
        dt = (time.time() - t0) * 1000.0

        det_array = self._to_detection2darray(results, msg)
        self.pub.publish(det_array)

        self.get_logger().debug(
            f"YOLO detections: {len(det_array.detections)} | {dt:.1f} ms"
        )

    def _to_detection2darray(self, results, img_msg: Image) -> Detection2DArray:
        """
        Ultralytics results → vision_msgs/Detection2DArray
        """
        out = Detection2DArray()
        out.header = img_msg.header

        if not results:
            return out

        res = results[0]
        # res.boxes.xyxy, res.boxes.conf, res.boxes.cls
        if not hasattr(res, 'boxes') or res.boxes is None:
            return out

        h = img_msg.height
        w = img_msg.width

        xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, 'xyxy') else []
        confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, 'conf') else []
        clss  = res.boxes.cls.cpu().numpy()  if hasattr(res.boxes, 'cls')  else []

        for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss):
            det = Detection2D()
            det.header = img_msg.header

            # bounding box center/size
            cx = float((x1 + x2) / 2.0)
            cy = float((y1 + y2) / 2.0)
            bw = float(max(x2 - x1, 0.0))
            bh = float(max(y2 - y1, 0.0))

            det.bbox = BoundingBox2D()
            det.bbox.center.position.x = cx
            det.bbox.center.position.y = cy
            det.bbox.size_x = bw
            det.bbox.size_y = bh

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(int(cls_id))
            hyp.hypothesis.score = float(score)
            det.results.append(hyp)

            out.detections.append(det)

        return out


def main(args=None):
    rclpy.init(args=args)
    node = YoloSubscriber()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()