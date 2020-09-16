# Make sure the doctests in the docs can import the module
import sys
sys.path.insert(0, '.')

from hypothesis import settings

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


def pytest_configure(config):
    # Set Hypothesis max_examples.
    hypothesis_max_examples = config.getoption("--hypothesis-max-examples")
    if hypothesis_max_examples is not None:
        import hypothesis

        hypothesis.settings.register_profile(
            "ndindex-hypothesis-overridden", max_examples=int(hypothesis_max_examples)
        )

        hypothesis.settings.load_profile("ndindex-hypothesis-overridden")


settings.register_profile('ndindex-hypothesis-profile', deadline=800)
settings.load_profile('ndindex-hypothesis-profile')
