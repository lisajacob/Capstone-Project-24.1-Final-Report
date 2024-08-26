from setuptools import setup, find_packages

setup(
    name='moon_image_processor',
    version='0.1.0',  # Replace with your desired version
    description='A package for preprocessing moon images',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'patchify',
        'Pillow',
        'opencv-python',
        'numpy',
        'matplotlib',
        'seaborn'
        # Add any other dependencies from your requirements.txt
    ]
)