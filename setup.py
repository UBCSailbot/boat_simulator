from os.path import join
from glob import glob
from setuptools import setup, find_packages

PACKAGE_NAME = 'boat_simulator'
REQUIRED_MODULES = [
    'setuptools',
    'numpy',
    'scipy'
]

setup(
    name=PACKAGE_NAME,
    version='0.0.0',
    packages=find_packages(exclude=["tests"]),
    install_requires=REQUIRED_MODULES,
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + PACKAGE_NAME]
        ),
        (
            join('share', PACKAGE_NAME),
            glob('launch/*launch.[pxy][yma]*')
        ),
        (
            join('share', PACKAGE_NAME),
            ['package.xml']
        ),
    ],
    zip_safe=True,
    maintainer='Devon Friend',
    maintainer_email='software@ubcsailbot.org',
    description='UBC Sailbot\'s Boat Simulator',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'physics_engine_node = ' +
            'boat_simulator.nodes.physics_engine.physics_engine_node:main',
        ],
    },
)
