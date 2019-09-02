from setuptools import setup, find_packages


with open('requirements.txt', 'r') as fh:
    requirements = fh.read().splitlines()

setup(
    name='mlkit',
    version='1.0.0',
    use_scm_version=False,
    url='https://github.com/bonobos/mlkit',
    packages=find_packages(),
    include_package_data=True,
    # setup_requires=['setuptools_scm', 'pytest-runner'],
    tests_require=['pytest'],
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3'
    ]
)
