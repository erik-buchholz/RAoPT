# ------------------------------------------------------------------------------
#  Author: Erik Buchholz
#  E-mail: e.buchholz@unsw.edu.au
# ------------------------------------------------------------------------------

from setuptools import setup

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

with open('LICENCE', 'r') as f:
    LICENCE_TEXT = f.read()


setup(
    name='RAoPT: Reconstruction Attack on Protected Trajectories',
    version='1.0.0',
    packages=['test', 'raopt'],
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    license=LICENCE_TEXT,
    author='Erik Buchholz',
    author_email='e.buchholz@unsw.edu.au',
    description='Reconstruction Attack on Differential Private Trajectory Protection Mechanisms (RAoPT)',
    python_requires='>=3.9'
)
