from setuptools import setup

package_name = 'my_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/all_nodes.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/yolo_viz.rviz']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Yu Jung Yi',
    maintainer_email='lee911230@gmail.com',
    description='Bringup launch files',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [],
    },
)
