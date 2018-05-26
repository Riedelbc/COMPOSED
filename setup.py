from setuptools import setup

setup(
    name='composed',
    version='0.1',
    packages=['composed'],
    url='',
    license='proprietary',
    author='Gautam Prasad',
    author_email='gprasad@usc.edu',
    description='Runs the Classification Optimization with Merged Partitions Over SEx and Disease (COMPOSED)'
                'algorithm by Brandy Lyjak, inspired by the EPIC algorithm by Gautam Prasad.',
    install_requires=["numpy",
                      "scikit-learn",
                      "scipy",
                      "sphinx",
                      "statsmodels",
                      "matplotlib",
                      "pandas"],
)
