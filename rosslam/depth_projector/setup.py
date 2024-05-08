from setuptools import find_packages, setup

package_name = 'depth_projector'

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
    maintainer='emperornao',
    maintainer_email='ansicpp2020@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_pub_1      = camera_setup.camera_pub_1:main',
            'camera_pub_2      = camera_setup.camera_pub_2:main',
            'disparity_publisher = depth_projector.disparity_publisher:main',
            'depth_publisher = depth_projector.mock_depth_publisher:main',
            'projector = depth_projector.projector:main'
        ],
    },
)
