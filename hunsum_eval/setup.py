from setuptools import setup

setup(
    name='hunsum-eval',
    version='',
    packages=['tests', 'errors', 'metrics', 'results', 'entrypoints'],
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
