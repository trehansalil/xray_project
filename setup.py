from setuptools import find_packages, setup
from typing import List
from xray.constants import *
import os

requirements_path = os.path.join(os.getcwd(), requirement_file_name)

def get_requirements(file_path:str) ->List[str]:
    requirements=[]
    HYPEN_E_DOT = '-e .'
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", '') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    print(requirements)
    return requirements

setup(
    name='xray-classification',
    version='0.0.1',
    author='Salil Trehan',
    author_email='trehansalil1@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(file_path=requirements_path),
)