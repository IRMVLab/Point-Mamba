import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = '1.1.0'

with open("README.md", "r", encoding="utf-8") as fid:
  long_description = fid.read()


dir = './csrc'
sources = ['{}/{}'.format(dir, src) for src in os.listdir(dir)
           if src.endswith('.cpp') or src.endswith('.cu')]

setup(
    name='dwconv',
    version=__version__,
    author='Peng-Shuai Wang',
    author_email='wangps@hotmail.com',
    url='https://github.com/octree-nn/dwconv',
    description='Octree-based Depth-Wise Convolution',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['dwconv'],
    # packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=['torch', 'numpy', 'ocnn'],
    python_requires='>=3.8',
    license='MIT',
    ext_modules=[
        CUDAExtension(
            name='dwconv.core',
            sources=sources)
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        'License :: OSI Approved :: MIT License',
    ],
)
