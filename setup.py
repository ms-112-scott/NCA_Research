from setuptools import setup, find_packages

setup(
    name="my_nca_core",  # 幫你的套件取個名字
    version="0.1",
    packages=find_packages(),  # 自動尋找有 __init__.py 的資料夾
)
