from setuptools import setup, find_packages

setup(
   name='Final Project',
   version='0.0.1',
   description='Final Project from Udacity',
   author='Fabio Barbazza',
   packages=find_packages(),
   install_requires=['joblib==1.0.1',
                    'pandas==1.3.5',
                    'pytest==6.2.5'] #external packages as dependencies
)