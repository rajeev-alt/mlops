from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirement(file_path: str) -> List:
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [ req.replace("\n", "") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
        return requirements


setup(
    name='mlops',
    version='0.1',
    packages=find_packages(),
    license='BSD Lisence',
    author='Rajiv Kumar',
    author_email='rajiv@sigmoidanalytics.com',
    description='This package and project is base frameworks of data science ',
    install_requires=get_requirement("requirement.txt"),
)