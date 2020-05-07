from setuptools import setup

with open('README.md', 'r') as f:
    readme = f.read()


with open('requirements.txt', 'r') as f:
    requirements = f.read()


entry_points = {'console_scripts': [
    'turboparser = turboparser.scripts.run_parser:main',
    'turboparser-train = turboparser.scripts.train_parser:main',
]}

setup(
    name='turboparser',
    version='0.1.0',
    description='Python port of the Turbo Parser toolkit',
    long_description=readme,
    author='André Martins and Erick Fonseca',
    author_email='erickrfonseca@gmail.com',
    license='LGPL',
    packages=['turboparser', 'turboparser.layers', 'turboparser.parser',
              'turboparser.commons', 'turboparser.scripts'],
    entry_points=entry_points,
    install_requires=requirements
)
