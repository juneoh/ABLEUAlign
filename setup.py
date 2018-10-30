from setuptools import setup, find_packages

setup(name='ableualign',
      version='0.1.1',
      description='Sentence alignment using Advanced BLEU metrics',
      url='https://github.com/juneoh/ableualign',
      author='June Oh',
      license='MIT',
      classifiers=[
          'Topic :: Text Processing',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      python_requires='>=3.6.0', packages=find_packages(),
      entry_points={
          'console_scripts': ['ableualign=ableualign.__main__:main'],
      },
      include_package_data=True)
