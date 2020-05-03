"""paver config file"""

# from testing python book
from paver.easy import sh
from paver.tasks import task, needs


@task
def tests():
    """unit testing"""
    # sh('nosetests --verbose --cover-package=techminer --cover-tests '
    #   ' --with-doctest --rednose  ./techminer/')
    sh("pytest ")


@task
def pylint():
    """pyltin"""
    sh("pylint ./techminer/")


@task
def pypi():
    """Instalation on PyPi"""
    sh("python setup.py sdist")
    sh("twine upload dist/*")


@task
def local():
    """local install"""
    sh("pip3 uninstall techminer")
    sh("python3 setup.py install develop")


@task
def sphinx():
    """Document creation using Shinx"""
    sh("cd guide; make html; cd ..")


@needs("nosetests", "pylint", "sphinx")
@task
def default():
    """default"""
    pass
