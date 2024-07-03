from setuptools import setup, find_packages

setup(
    name="preprintscout",
    version="0.1.4",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "arxiv>=2.1.3",
        "beautifulsoup4>=4.12.3",
        "google-generativeai>=0.7.0",
        "langdetect>=1.0.9",
        "openai>=1.35.7",
        "pandas>=2.2.2",
        "pytz>=2024.1",
        "Requests>=2.32.3",
        "retry>=0.9.2",
        "scikit_learn>=1.4.2",
    ],
)
