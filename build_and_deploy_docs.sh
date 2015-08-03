#!/bin/bash

bash docs/convert_nbs_to_md.sh
mkdocs build --clean
mkdocs gh-deploy
