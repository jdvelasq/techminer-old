from setuptools import setup


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='techminer',
      version='0.0.0',
      description='Tech mining of bibliograpy',
      long_description='Tech mining of bibliograpy',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
      ],
      keywords='bibliograpy',
      url='http://github.com/jdvelasq/techMiner',
      author='Juan D. Velasquez & Ivanohe J. Garces',
      author_email='jdvelasq@unal.edu.co',
      license='MIT',
      packages=['techminer'],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)
