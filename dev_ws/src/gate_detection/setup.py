from setuptools import setup

package_name = 'gate_detection'

setup(
    name=package_name,
    version='0.6.3',
    packages=[package_name],
    
    install_requires=['setuptools'],
    zip_safe=True,
    author='Mikael Arguedas',
    author_email='mikael@osrfoundation.org',
    maintainer='Mikael Arguedas',
    maintainer_email='mikael@osrfoundation.org',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Examples of minimal publishers using rclpy.',
    license='Apache License, Version 2.0',
    entry_points={
        'console_scripts': [
            'talker ='
            'gate_detection.simple_publisher:main',
        ],
    },
)
