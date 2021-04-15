import nox
import nox_poetry

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["tests", "mypy", "lint"]


@nox_poetry.session(python=["3.7", "3.8"])
def tests(session):
    """ Run test suit."""
    args = session.posargs or ["--cov"]
    session.install("pytest", "pytest-cov", ".")
    session.run("pytest", *args)


@nox_poetry.session(python="3.8")
def lint(session):
    """ Lint using flake8."""
    args = session.posargs or []
    session.install(
        "flake8", "flake8-bugbear", "flake8-black", "flake8-isort",
    )
    session.run("flake8", *args)


@nox_poetry.session(python="3.8")
def mypy(session):
    """ Type-check using mypy."""
    args = session.posargs or ["src"]
    session.install("mypy", ".")
    session.run("mypy", *args)


@nox_poetry.session(python="3.8")
def codecov(session):
    """ Upload coverage data to Codecov"""
    session.install("codecov", "coverage")
    session.run("coverage", "xml")
    session.run("codecov", *session.posargs)
