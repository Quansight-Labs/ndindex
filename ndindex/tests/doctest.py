"""
Custom script to run the doctests

This runs the doctests but ignores trailing ``` in Markdown documents.

Running this separately from pytest also allows us to not include the doctests
in the coverage. It also allows us to force a separate namespace for each
docstring's doctest, which the pytest doctest integration does not allow.

TODO: Make these tests also run with pytest, but still keeping them out of the
coverage.

"""

import sys
import unittest
import glob
import os
from contextlib import contextmanager
from doctest import DocTestRunner, DocFileSuite, DocTestSuite, NORMALIZE_WHITESPACE

@contextmanager
def patch_doctest():
    """
    Context manager to patch the doctester

    The doctester must be patched
    """
    orig_run = DocTestRunner.run

    def run(self, test, **kwargs):
        for example in test.examples:
            # Remove ```
            example.want = example.want.replace('```\n', '')
            example.exc_msg = example.exc_msg and example.exc_msg.replace('```\n', '')

        return orig_run(self, test, **kwargs)

    try:
        DocTestRunner.run = run
        yield
    finally:
        DocTestRunner.run = orig_run

DOCS = os.path.realpath(os.path.join(__file__, os.path.pardir, os.path.pardir,
                                     os.pardir, 'docs'))
MARKDOWN = glob.glob(os.path.join(DOCS, '**', '*.md'), recursive=True)
RST = glob.glob(os.path.join(DOCS, '**', '*.rst'), recursive=True)
README = os.path.realpath(os.path.join(__file__, os.path.pardir, os.path.pardir,
                                     os.pardir, 'README.md'))
def load_tests(loader, tests, ignore):
    for mod in sys.modules:
        if mod.startswith('ndindex'):
            # globs={} makes the doctests not include module names
            tests.addTests(DocTestSuite(sys.modules[mod], globs={},
                                        optionflags=NORMALIZE_WHITESPACE))
    tests.addTests(DocFileSuite(*MARKDOWN, *RST, README,
                                optionflags=NORMALIZE_WHITESPACE,
                                module_relative=False))
    return tests

def doctest():
    with patch_doctest():
        return unittest.main(module='ndindex.tests.doctest', exit=False).result

if __name__ == '__main__':
    # TODO: Allow specifying which doctests to run at the command line
    doctest()
