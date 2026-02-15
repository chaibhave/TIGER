from setuptools import setup, find_packages

setup(
    name='TIGER',
    version='1.0.0',
    description='Tool for Interactive Generation of Exodus Renders',
    author='Chaitanya Bhave',
    author_email='chaitanya.bhave@gatech.edu',
    url='https://github.com/chaitanyaBhave26/TIGER',
    py_modules=['ExodusReader'],
    packages=find_packages(),
    install_requires=[
        'matplotlib>=3.5',
        'numpy>=1.20',
        'scipy>=1.7',
        'h5py',
        'netCDF4>=1.6.4',
        'mpi4py',
        'pytest',
        'cmcrameri>=1.3',
    ],
    extras_require={
        'video': ['opencv-python'],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)