from rever.activity import activity

@activity
def run_tests():
    # Don't use the built-in pytest action because that uses Docker, which is
    # overkill and requires installing Docker
    pytest

$PROJECT = 'ndindex'
$ACTIVITIES = [
    'run_tests',
    'tag',  # Creates a tag for the new version number
    'push_tag',  # Pushes the tag up to the $TAG_REMOTE
    'pypi',  # Sends the package to pypi
    'conda_forge',  # Creates a PR into your package's feedstock
    'ghrelease'  # Creates a Github release entry for the new tag
]

$PUSH_TAG_REMOTE = 'git@github.com:Quansight/ndindex.git'  # Repo to push tags to

$GITHUB_ORG = 'Quansight'  # Github org for Github releases and conda-forge
$GITHUB_REPO = 'ndindex'  # Github repo for Github releases and conda-forge
