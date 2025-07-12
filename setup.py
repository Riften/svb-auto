from setuptools import setup, find_packages


setup(
    name="svb-auto-duel",
    version="1.2.2",
    author="",
    author_email="",
    description="影之诗自动战斗工具 - 基于机器视觉的Android模拟器自动化",
    long_description_content_type="text/markdown",
    url="https://github.com/yongxi/SVBYD_autoduel",
    package_dir={"": "python_package"},
    packages=find_packages(where="python_package"),
    python_requires=">=3.7",
)