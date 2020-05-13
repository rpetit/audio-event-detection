from setuptools import setup, find_packages

setup(name='evdetect',
      packages=find_packages(),
      description='Audio event detection library',
      url='https://github.com/rpetit/audio-event-detection',
      install_requires=['numpy', 'scipy', 'matplotlib', 'librosa'])
