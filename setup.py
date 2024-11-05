from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r', encoding='utf-8') as fh:
    install_requires = fh.read().splitlines()

setup(
    name='outetts',
    version='0.1.5',
    packages=find_packages(),
    install_requires=install_requires,
    author='OuteAI',
    description='OuteAI Text-to-Speech (TTS)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/OuteAI/OuteTTS',
    package_data={
        "outetts.v0_1": ["wavtokenizer_config.yaml"],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
