#!/bin/bash
pip install black
pip install autoflake
find . -maxdepth 1 -name "*.py" -exec bash -c "echo {}; autoflake -v -cd --in-place --remove-unused-variables --remove-all-unused-imports {}; black {}" \;
