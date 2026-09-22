import glob
import os

from setuptools import find_packages, setup

package_name = 'dgl_ros_pipeline'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob.glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob.glob('config/*.rviz')),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Tecsi Zsuzsanna Vilma',
    maintainer_email='tecsizsv@gmail.com',
    description='Visualization and machine vision for DGL in ROS',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
          'grasp_client = dgl_ros_pipeline.grasp_client:main',
        ],
    },
)
