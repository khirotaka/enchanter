import os
from setuptools import setup, find_packages

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


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='enchanter',
    version='0.5.3.1',
    packages=find_packages(exclude=["examples", "tests"]),
    url='https://github.com/khirotaka/enchanter',
    keywords="pytorch comet_ml",
    license='Apache-2.0',
    author='Hirotaka Kawashima',
    author_email='',
    description='Enchanter is a library for machine learning tasks for comet.ml users.',
    install_requires=load_requirements(),
    long_description=long_description,
    long_description_content_type='text/markdown'
)
