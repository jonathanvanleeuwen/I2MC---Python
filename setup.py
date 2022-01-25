import setuptools
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('src/I2MC/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setuptools.setup(
    name=main_ns['__title__'],
    version=main_ns['__version__'],
    author=main_ns['__author__'],
    author_email=main_ns['__email__'],
    description=main_ns['__description__'],
    long_description=readme,
    long_description_content_type="text/markdown",
    url=main_ns['__url__'],
    project_urls={
        "Source Code": main_ns['__url__'],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license=license,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=['numpy','matplotlib','scipy','pandas']
)
