from setuptools import setup, find_packages

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fh:
    requirements = fh.read().splitlines()

setup(
    name='mlkit',
    author='Diego De Lazzari',
    author_email='ddelazzari@bonobos.com',
    version='1.0.1',
    use_scm_version=False,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/bonobos/mlkit',
    packages=find_packages(),
    include_package_data=False,
    tests_require=['pytest'],
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python :: 3'
    ]
)
