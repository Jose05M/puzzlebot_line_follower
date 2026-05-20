import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'puzzlebot_line_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jos',
    maintainer_email='jos@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'traffic_light_detector = puzzlebot_line_follower.traffic_light_detector:main',
            'line_follower_controller  = puzzlebot_line_follower.line_follower_controller:main',
            'puzzlebot_odometry  = puzzlebot_line_follower.puzzlebot_odometry:main',
            'line_detector =  puzzlebot_line_follower.line_detector:main',
            'follower_node =  puzzlebot_line_follower.follower_node:main',
            'line_det =  puzzlebot_line_follower.line_det:main',
        ],
    },
)
