from setuptools import setup

setup(
    name='enchanter',
    version='0.1',
    packages=['enchanter', 'enchanter.nn', 'enchanter.estimator'],
    url='https://github.com/khirotaka/enchanter',
    license='Apache-2.0',
    author='Hirotaka Kawashima',
    author_email='',
    description='Machine Learning Pipeline, Training and Logging for Me.',
    install_requires=['torch', 'numpy', 'scikit-learn', 'tqdm', 'torchvision', 'comet-ml']
)
