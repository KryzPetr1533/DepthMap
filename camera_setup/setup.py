from setuptools import setup

package_name = 'camera_setup'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hcx',
    maintainer_email='hcx@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
         'camera_pub_1      = camera_setup.camera_pub_1:main',
         'camera_pub_2      = camera_setup.camera_pub_2:main',
         'camera_sub      = camera_setup.camera_sub:main',
        ],
    },
)
