from setuptools import setup, find_packages

setup(
    name='hunsum-eval',
    version='1.0',
    packages=find_packages(
        # All keyword arguments below are optional:
        where='hunsum_eval',  # '.' by default
        # include=['mypackage*'],  # ['*'] by default
        # exclude=['mypackage.tests'],  # empty by default
    ),
    package_dir={'': 'hunsum_eval/tests'},
    url='',
    license='',
    author='Dorina Lakatos',
    author_email='',
    description='',
    install_requires=[
        'summ-eval'
    ]
)
