import setuptools, versioneer

with open("README.rst", "r") as fh:
    long_description = fh.read()
with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="doodleverse-utils",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Daniel Buscombe",
    author_email="dbuscombe@gmail.email",
    description="Imports into the Doodleverse",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
)
