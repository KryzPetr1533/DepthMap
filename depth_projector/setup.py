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
    install_requires=[
        'setuptools',
        'opencv-python',  # Ensure all required packages are listed
        'cv_bridge',
        'numpy',
        'pycuda',
        'tensorrt-cu12',
    ],
    zip_safe=True,
    maintainer='devExplorer',
    maintainer_email='pkkryzh1533@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "camera_pub = depth_projector.camera_pub:main",
            "depth_publisher = depth_projector.depth_publisher:main",
            "projector = depth_projector.projector:main"
        ],
        
    },
)
