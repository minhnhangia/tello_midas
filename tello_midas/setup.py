from setuptools import find_packages, setup

package_name = 'tello_midas'

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
    maintainer='minh',
    maintainer_email='minhnhangia@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "midas_inference = tello_midas.midas_inference:main",
            "midas_analysis = tello_midas.midas_analysis:main",
            "multi_midas_inference = tello_midas.multi_midas_inference:main",
        ],
    },
)
