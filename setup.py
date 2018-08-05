from setuptools import setup, find_packages


readme = 'TODO: write longer description'
license = 'MIT'

setup(
    name='seqNMF',
    version='0.1.0',
    description='Python implementation of seqNMF',
    long_description=readme,
    author='Contextual Dynamics Laboratory',
    author_email='contextualdynamics@gmail.com',
    url='https://www.context-lab.com',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

