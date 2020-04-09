from tempfile import mkdtemp

from rever.activity import activity
from rever.conda import run_in_conda_env

@activity
def mktmp():
    curdir = $(pwd).strip()
    tmpdir = mkdtemp(prefix=$GITHUB_REPO + "_")
    print(f"Running in {tmpdir}")
    cd @(tmpdir)
    git clone @(curdir)
    cd $GITHUB_REPO

@activity
def run_tests():
    # Don't use the built-in pytest action because that uses Docker, which is
    # overkill and requires installing Docker
    with run_in_conda_env(['python=3.8', 'pytest', 'numpy', 'hypothesis', 'pyflakes']):
        pyflakes .
        python -We:invalid -We::SyntaxWarning -m compileall -f -q ndindex/
        pytest

@activity
def build_docs():
    with run_in_conda_env(['python=3.8', 'sphinx', 'recommonmark']):
        cd docs
        make html

@activity
def annotated_tag():
    # https://github.com/regro/rever/issues/212
    git tag -a -m "$GITHUB_REPO $VERSION release" $VERSION

$PROJECT = 'ndindex'
$ACTIVITIES = [
    # 'mktmp',
    'run_tests',
    'build_docs',
    'annotated_tag',  # Creates a tag for the new version number
    'push_tag',  # Pushes the tag up to the $TAG_REMOTE
    'pypi',  # Sends the package to pypi
    'ghrelease',  # Creates a Github release entry for the new tag
    'conda_forge',  # Creates a PR into your package's feedstock
]

$PUSH_TAG_REMOTE = 'git@github.com:Quansight/ndindex.git'  # Repo to push tags to

$GITHUB_ORG = 'Quansight'  # Github org for Github releases and conda-forge
$GITHUB_REPO = 'ndindex'  # Github repo for Github releases and conda-forge
