import os
from setuptools import setup

PATH_ROOT = os.path.dirname(__file__)


def load_requirements(path_dir=PATH_ROOT, comment_char='#'):
    with open(os.path.join(path_dir, 'requirements.txt'), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)]
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


setup(
    name='enchanter',
    version='0.3',
    packages=[
        'enchanter', 'enchanter.addons',
        'enchanter.engine', 'enchanter.ensemble',
        'enchanter.metrics', 'enchanter.wrappers'
    ],
    url='https://github.com/khirotaka/enchanter',
    license='Apache-2.0',
    author='Hirotaka Kawashima',
    author_email='',
    description='Machine Learning Pipeline, Training and Logging for Me.',
    install_requires=load_requirements()
)
