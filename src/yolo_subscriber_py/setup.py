from setuptools import find_packages, setup

package_name = 'yolo_subscriber_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yu Jung Yi',
    maintainer_email='lee911230@gmail.com',
    description='YOLO subscriber node',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_subscriber_py_node = yolo_subscriber_py.yolo_subscriber_py_node:main',
            'det_to_pc_node = yolo_subscriber_py.det_to_pc_node:main',
        ],
    },
)
