from setuptools import setup

package_name = 'image_enhancement'

setup(
    name=package_name,
    version='0.6.3',
    packages=[package_name],
    
    install_requires=['setuptools'],
    zip_safe=True,
    author='Gad Mohamed',
    author_email='gadm43@yahoo.com',
    
    description='subscribing to raw frames, process with GANS, and publish to object detection and target analysis nodes.',
    entry_points={
        'console_scripts': [
            'enhance ='
            'image_enhancement.enhancer:main'
        ],
    },
)
