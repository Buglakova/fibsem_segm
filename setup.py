from setuptools import setup, find_packages

setup(
    name='cryofib',
    version='1.1',
    description='Image analysis for correlative fluorescence - FIB-SEM - cryo pipeline.',
    url='https://github.com/Buglakova/fibsem_segm',
    packages=find_packages(include=['cryofib']),
    python_requires='>=3.6',
    install_requires=[],
    author='Elena Buglakova',
    author_email='elena.buglakova@embl.de',
    license='MIT'
)
