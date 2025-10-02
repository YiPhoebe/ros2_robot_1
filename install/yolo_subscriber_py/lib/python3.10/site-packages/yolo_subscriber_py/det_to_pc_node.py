#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2


class DetectionsToPointCloud(Node):
    def __init__(self):
        super().__init__('det_to_pc_node')

        # Parameters
        self.declare_parameter('detections_topic', '/yolo/bounding_boxes')
        self.declare_parameter('pc_topic', '/yolo/centers')
        self.declare_parameter('frame_id', 'camera')

        detections_topic = self.get_parameter('detections_topic').value
        pc_topic = self.get_parameter('pc_topic').value

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.sub = self.create_subscription(
            Detection2DArray, detections_topic, self.cb_dets, qos
        )
        self.pub = self.create_publisher(PointCloud2, pc_topic, 10)

        self.get_logger().info(
            f"Subscribing: {detections_topic} → Publishing: {pc_topic} (PointCloud2)"
        )

    def cb_dets(self, msg: Detection2DArray):
        # Convert Detection2DArray to PointCloud2: use bbox centers as x,y; z=0; intensity=score
        points = []
        for det in msg.detections:
            cx = float(det.bbox.center.position.x)
            cy = float(det.bbox.center.position.y)
            # pick highest score if multiple hypotheses
            score = 0.0
            if det.results:
                score = max(float(h.hypothesis.score) for h in det.results)
            points.append((cx, cy, 0.0, score))

        header = msg.header
        if not header.frame_id:
            header.frame_id = self.get_parameter('frame_id').value

        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        cloud = pc2.create_cloud(header, fields, points)
        self.pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionsToPointCloud()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

