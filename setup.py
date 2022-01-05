from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

readme = 'JAX implementation of seqNMF.  For more information visit https://github.com/calebweinreb/seqnmf-gpu.'

with open('LICENSE') as f:
    license = f.read()

setup(
    name='seqnmf',
    description='JAX implementation of seqNMF',
    long_description=readme,
    author='Caleb Weinreb',
    author_email='calebsw@gmail.com',
    url='https://github.com/calebweinreb/seqnmf-gpu',
    license=license,
    install_requires=requirements,
    package_data = {'seqnmf':['data/MackeviciusData.mat']},
    packages=find_packages(exclude=('tests', 'docs'))
)
