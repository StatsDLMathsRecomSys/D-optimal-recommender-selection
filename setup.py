from setuptools import setup, find_packages

print(find_packages())
setup(name='uni_bandit',
      version='1.0',
      packages=['uni_bandit'],
      package_data={"uni_bandit": ["py.typed"]})