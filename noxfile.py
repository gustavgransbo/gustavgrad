import nox

nox.options.reuse_existing_virtualenvs = True


@nox.session(python=["3.7", "3.8"])
def tests(session):
    """ Run test suit."""
    args = session.posargs or ["--cov"]
    session.run("poetry", "install", external=True)
    session.run("pytest", *args)


@nox.session(python="3.8")
def lint(session):
    """ Lint using flake8."""
    args = session.posargs or []
    session.run("poetry", "install", external=True)
    session.run("flake8", *args)


@nox.session(python="3.8")
def mypy(session):
    """ Type-check using mypy."""
    args = session.posargs or ["src"]
    session.run("poetry", "install", external=True)
    session.run("mypy", *args)
