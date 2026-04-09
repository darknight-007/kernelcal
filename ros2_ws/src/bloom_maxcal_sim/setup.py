from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'bloom_maxcal_sim'

setup(
    name=package_name,
    version='0.1.0',
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
    maintainer='Jnaneshwar Das',
    maintainer_email='jnaneshwar.das@asu.edu',
    description=(
        'MaxCal-based algae bloom following with simulated 2D rover. '
        'Implements the adaptive algal-bloom scenario from arXiv:2603.27880.'
    ),
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            # Individual nodes
            'bloom_field_node = bloom_maxcal_sim.nodes.bloom_field_node:main',
            'rover_sim_node   = bloom_maxcal_sim.nodes.rover_sim_node:main',
            'maxcal_controller_node = bloom_maxcal_sim.nodes.maxcal_controller_node:main',
            'visualizer_node  = bloom_maxcal_sim.nodes.visualizer_node:main',
        ],
    },
)
