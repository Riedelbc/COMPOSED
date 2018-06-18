from setuptools import setup

setup(
    name='composed',
    version='0.1',
    packages=['composed'],
    url='',
    license='proprietary',
    author='Brandy Riedel',
    author_email='riedelbc@gmail.com',
    description='Runs the Classification Optimization with Merged Partitions Over SEx and Disease (COMPOSED)'
                'algorithm by Brandy Riedel, inspired by the EPIC algorithm by Gautam Prasad.',
    install_requires=["numpy",
                      "scikit-learn",
                      "scipy",
                      "sphinx",
                      "statsmodels",
                      "matplotlib",
                      "pandas"],
)
