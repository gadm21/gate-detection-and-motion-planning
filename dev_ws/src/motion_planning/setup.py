from setuptools import setup

package_name = 'motion_planning'

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
    description='Examples of minimal subscribers using rclpy.',
    license='Apache License, Version 2.0',
    
    entry_points={
        'console_scripts': [
            
            'listener ='
            'motion_planning.simple_subscriber:main',
        ],
    },
)
