from setuptools import setup, find_packages

setup(name='evdetect',
      packages=find_packages(),
      description='Audio event detection library',
      url='https://github.com/romainpe/audio-event-detection',
      install_requires=['numpy', 'scipy', 'librosa'],
      python_requires='>=3.5')
