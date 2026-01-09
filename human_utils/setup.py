from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'human_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nvidia',
    maintainer_email='nvidia@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hand_wave_detection_node = human_utils.handwave_by_jw:main',
            
            'human_state_detector_node = human_utils.human_state_detector:main',
            'hand_up_goal_monitor_node = human_utils.hand_up_goal_monitor:main',
            'hand_up_goal_controller_node = human_utils.hand_up_goal_controller:main',
            'save_current_pose_node = human_utils.save_current_pose:main',
        ],
    },
)
