from setuptools import setup

with open('README.md', 'r') as f:
    readme = f.read()

setup(
    name='turboparser',
    version='0.1.0',
    description='Python port of the Turbo Parser toolkit',
    long_description=readme,
    author='André Martins and Erick Fonseca',
    author_email='erickrfonseca@gmail.com',
    license='LGPL',
    packages=['turboparser', 'turboparser.layers', 'turboparser.parser',
              'turboparser.commons']
)
