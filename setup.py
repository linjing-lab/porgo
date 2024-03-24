import re
import sys

from setuptools import setup

if sys.version_info < (3, 7, 0):
    raise OSError(f'porgo requires Python >=3.7, but yours is {sys.version}')

VERSION_FILE = 'porgo/_version.py'
VERSION_REGEXP = r'^__version__ = \'(\d+\.\d+\.\d+)\''

r = re.search(VERSION_REGEXP, open(VERSION_FILE).read(), re.M)
if r is None:
    raise RuntimeError(f'Unable to find version string in {VERSION_FILE}.')

version = r.group(1)

try:
    with open('README.md', 'r', encoding='utf-8') as fp:
        _long_description = fp.read()
except FileNotFoundError:
    _long_description = ''

setup(
      name='porgo',  # pkg_name
      packages=['porgo',],
      version=version,  # version number
      description="The portable universal library in global optimization.",
      author='林景',
      author_email='linjing010729@163.com',
      license='MIT',
      url='https://github.com/linjing-lab/porgo',
      download_url='https://github.com/linjing-lab/porgo/tags',
      long_description=_long_description,
      long_description_content_type='text/markdown',
      include_package_data=True,
      zip_safe=False,
      setup_requires=['setuptools>=18.0', 'wheel'],
      project_urls={
            'Source': 'https://github.com/linjing-lab/porgo/tree/master/porgo/',
            'Tracker': 'https://github.com/linjing-lab/porgo/issues',
      },
      classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'License :: OSI Approved :: MIT License',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
      ],
      install_requires=[
            'numpy>=1.21.0',
      ],
      # extras_require=[]
)
