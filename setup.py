from setuptools import setup, find_packages

setup(
    name='nlp_utils',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[

    ],
    author='Karim Dabbouz',
    author_email='hey@karim.ooo',
    description='A Python package with utils for different text analysis tasks in German.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/karimdabbouz/nlp_utils',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
