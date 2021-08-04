#!/bin/bash
# Run and update the benchmarks in the gh-pages branch. The resulting
# benchmarks are served at
# https://quansight-labs.github.io/ndindex/benchmarks/index.html
set -e
set -x

asv run -k ALL
asv publish
git checkout gh-pages
git pull
rm -rf benchmarks
cp -R .asv/html benchmarks
git add benchmarks
git commit -m "Update benchmarks"
git checkout -
git push origin gh-pages
