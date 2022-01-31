from setuptools import setup, find_packages

setup(
   name='Final Project',
   version='0.0.1',
   description='Final Project from Udacity',
   author='Fabio Barbazza',
   packages=find_packages(),
   install_requires=['joblib==1.0.1',
                    'pandas==1.3.5',
                    'pytest==6.2.5',
                    'statsmodels==0.13.1',
                    'sklearn==0.0',
                    'dvc==2.9.3',
                    'boto3==1.20.24',
                    's3fs==2022.1.0',
                    'fastapi==0.72.0',
                    'uvicorn==0.17.0'
                    ] #external packages as dependencies
)