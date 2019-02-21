#! /bin/bash

mkdir tmpInstallHtMouse
cp environment.yml tmpInstallHtMouse

# * This is to check for debugging that running the conda installer twice gives no errors, delete after
cp /Users/chaitanyakatpatal/Downloads/Anaconda2-2018.12-MacOSX-x86_64.sh ./tmpInstallHtMouse/miniconda_installer

cd tmpInstallHtMouse

# TODO: for testing, we'll use anaconda since we already have it
# TODO: but change this back after we're done with whole thing
# * curl -fSL https://repo.anaconda.com/miniconda/Miniconda2-latest-MacOSX-x86_64.sh > miniconda_installer.sh

echo "\nMaking environment"
chmod +x miniconda_installer.sh
./miniconda_installer

conda env create --file=environment.yml

echo "\nRemoving Installation Files..." 
cd ..
rm -r tmpInstallHtMouse

echo "\nInstall completed successfully"