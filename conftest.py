# Make sure the doctests in the docs can import the module
import sys
sys.path.insert(0, '.')
try:
    from packaging.version import parse as LooseVersion
except ImportError:
    from distutils.version import LooseVersion

from hypothesis import settings

# Make sure a new enough version of NumPy is installed for the tests
import numpy
if LooseVersion(numpy.__version__) < LooseVersion('1.20'):
    raise RuntimeError("NumPy 1.20 (development version) or greater is required to run the ndindex tests")

# Show the NumPy version in the pytest header
def pytest_report_header(config):
    return f"project deps: numpy-{numpy.__version__}"

# Add a --hypothesis-max-examples flag to pytest. See
# https://github.com/HypothesisWorks/hypothesis/issues/2434#issuecomment-630309150

def pytest_addoption(parser):
    # Add an option to change the Hypothesis max_examples setting.
    parser.addoption(
        "--hypothesis-max-examples",
        "--max-examples",
        action="store",
        default=None,
        help="set the Hypothesis max_examples setting",
    )

    # Add an option to disable the Hypothesis deadline
    parser.addoption(
        "--hypothesis-disable-deadline",
        "--disable-deadline",
        action="store_true",
        help="disable the Hypothesis deadline",
    )


def pytest_configure(config):
    # Set Hypothesis max_examples.
    hypothesis_max_examples = config.getoption("--hypothesis-max-examples")
    disable_deadline = config.getoption('--hypothesis-disable-deadline')
    profile_settings = {}
    if hypothesis_max_examples is not None:
        profile_settings['max_examples'] = int(hypothesis_max_examples)
    if disable_deadline is not None:
        profile_settings['deadline'] = None
    if profile_settings:
        import hypothesis

        hypothesis.settings.register_profile(
            "ndindex-hypothesis-overridden", **profile_settings,
        )

        hypothesis.settings.load_profile("ndindex-hypothesis-overridden")


settings.register_profile('ndindex_hypothesis_profile', deadline=800)
settings.load_profile('ndindex_hypothesis_profile')
