from setuptools import setup
from setuptools import find_packages

with open('README.md', 'r') as f:
      long_description = f.read()

setup(
      author='Chamath Attanyaka',
      name='micronus',
      version='0.0.4',
      description='A light weight transformer model',
      packages=find_packages(include=['micronus', 'micronus.*']),
      install_requires=[
            'tensorflow==2.4.1',
            'keras==2.4.3',
            'numpy==1.19.3',
            'matplotlib>=3.3.2',
      ],
      python_requires='>=3.8',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      )
