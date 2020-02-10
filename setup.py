from setuptools import setup

setup(
    name='enchanter',
    version='0.2',
    packages=['enchanter', 'enchanter.addon', 'enchanter.engine'],
    url='https://github.com/khirotaka/enchanter',
    license='Apache-2.0',
    author='Hirotaka Kawashima',
    author_email='',
    description='Machine Learning Pipeline, Training and Logging for Me.',
    install_requires=['torch', 'numpy', 'scikit-learn', 'tqdm', 'torchvision']
)
