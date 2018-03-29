from setuptools import setup, find_packages

setup(name='evdetect',
      description='Audio event detection library',
      url='https://github.com/romainpe/audio-event-detection',
      install_requires=['numpy','scipy','librosa'],
      packages=find_packages())
