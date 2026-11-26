"""
*****************************************************
*               Color Tracker
*
*              Gabor Vecsei
* Website:     https://gaborvecsei.com
* Blog:        https://gaborvecsei.com
* LinkedIn:    https://www.linkedin.com/in/gaborvecsei
* Github:      https://github.com/gaborvecsei
*
*****************************************************
"""

from pathlib import Path

from setuptools import setup, find_packages

VERSION = '0.1.1'

root = Path(__file__).resolve().parent

long_description = (root / 'README.md').read_text(encoding='utf-8')

requirements = [
    line.strip()
    for line in (root / 'requirements.txt').read_text(encoding='utf-8').splitlines()
    if line.strip() and not line.lstrip().startswith('#')
]

setup(
    name='modern-color-tracker-ulaplappu',
    version=VERSION,
    description='Easy to use color tracking package for object tracking based on colors',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ulaplappu/ModernColorTracker',
    author='Gabor Vecsei',
    author_email='vecseigabor.x@gmail.com',
    license='MIT',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Topic :: Education',
        'Topic :: Software Development',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.13'],
    keywords='color tracker vecsei gaborvecsei color_tracker',
    install_requires=requirements,
    packages=find_packages(),
)
