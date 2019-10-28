#!/bin/sh
mkdir original_data
cd original_data
git init
git remote add origin -f git@github.com:hugo-alayrangues/touchstress.git
git config core.sparseCheckout true
echo "data/*" >> .git/info/sparse-checkout
git pull --depth=2 origin master
cd ..
