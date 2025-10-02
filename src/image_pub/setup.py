from setuptools import setup

package_name = 'image_pub'

setup(
    name='image-pub',
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='YOU',
    maintainer_email='YOU@example.com',
    description='Image publisher & recorder nodes',
    license='',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 퍼블리셔
            'image_pub_node = image_pub.image_pub_node:main',
            # 레코더(파일명에 맞춰서!)
            'image_recorder_node = image_pub.record_video_node:main',
            # RGB 비디오 퍼블리셔
            'image_pub_rgb_node = image_pub.image_pub_rgb:main',
        ],
    },
)