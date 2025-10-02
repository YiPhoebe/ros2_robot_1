from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node

def generate_launch_description():
    use_rviz = LaunchConfiguration('use_rviz')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_rviz', default_value='false',
            description='Start RViz2 to visualize topics'),
        # my_bringup/launch/all_nodes.launch.py 의 image_pub 노드
        Node(
            package='image_pub',
            executable='image_pub_node',
            name='image_pub',
            parameters=[{
                'use_dir': True,
                'dir_path': '/workspace/media/raw/Validation/01.원천데이터/VS',
                # JPG/JPEG 대소문자 모두 매칭
                'glob_pattern': '**/RGB-D(Depth)/*.png',
                'glob_recursive': True,
                'fps': 10.0,
                'topic': '/image_raw',
                'loop_dir': True,
            }]
        ),
        Node(
            package='yolo_subscriber_py',
            executable='yolo_subscriber_py_node',
            name='yolo_subscriber_py_node',
            output='screen',
            parameters=[
                {'model_path': '/workspace/models/yolov8n.pt'},   # 또는 yolov11n.pt
                {'conf': 0.25},
                {'imgsz': 640},
                {'every_n': 1},
                {'device': 'cpu'},
                {'topic_in': '/image_raw'},
                {'topic_out': '/yolo/bounding_boxes'}
            ]
        ),
        # Convert YOLO Detection2DArray to PointCloud2 for RViz
        Node(
            package='yolo_subscriber_py',
            executable='det_to_pc_node',
            name='det_to_pc',
            parameters=[
                {'detections_topic': '/yolo/bounding_boxes'},
                {'pc_topic': '/yolo/centers'},
                {'frame_id': 'camera'},
            ],
            output='screen',
        ),
        # RViz2 to visualize /image_raw and /yolo/centers
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/workspace/install/my_bringup/share/my_bringup/rviz/yolo_viz.rviz'],
            additional_env={'LIBGL_ALWAYS_SOFTWARE': '1', 'QT_QPA_PLATFORM':'xcb'},
            condition=IfCondition(use_rviz),
            output='screen',
        ),
        Node(
            package='overlay_viz',
            executable='overlay_viz_node',
            name='overlay',
            parameters=[{
                'input_topic': '/image_raw',
                'output_topic': '/image_overlay',
                'overlay_conf_min': 0.25,
            }],
        ),
        Node(
            package='image_pub',
            executable='image_recorder_node',
            name='recorder',
            parameters=[{
                'topic': '/image_overlay',                # 저장할 토픽
                'out_path': '/workspace/output/overlay.mp4',
                'fps': 10.0,
                'fourcc': 'mp4v',
            }],
        ),
    ])
