# Usage
Currently, this software is only tested on MacOS. That said, it should run well without configuration on both Windows and Linux devices with at least one working camera.

If you have more than one camera attached, you may need to change the camera device index to the one you want to use. This can be done by passing the -c flag

```bash
$ conda activate ht_mouse 
$ python ht_mouse.py # OPTIONAL: -c={Camera Device Index}
```

If you are running on MacOS Mojave (Version 10.14 or later) then you will be prompted to allow access to Accessibility Settings. This is to allow python the access to move the mouse around, so denying this will break the program.

# Installation
## Prerequisites

First download this repo to your local machine.
``` bash
git clone https://github.com/chai-kat/hand-tracking-mouse.git
```

This installation requires you to have Python 2.7, OpenCV 3, Numpy, TensorFlow and Pyautogui installed on your device. Installation of these is detailed below. 

If you do not have conda installed, then you will find it easiest to just use the install script provided. This downloads conda, and after giving you the option to accept the T&Cs, installs conda, and creates the "ht_mouse" environment with the necessary packages for running the program. This can be done by:

### On MacOS and Linux
```bash
$ ./installLinux.sh
```

### On Windows
``` cmd
$ installWindows.bat
```
If you have conda installed to a directory that's not the default, don't want to use conda, or want to use a GPU-compatible/specially built TensorFlow package, read on.


### With Conda (Any OS)

Create a conda environment with the required packages:
``` bash
$ conda env create -f environment.yml
```
Done. Easy as pie.

### With Pip (Also Any OS)
Install pyautogui and its dependencies:
``` bash
$ pip install -y pyautogui==0.9.38 
```
Install OpenCV and its dependencies:
```bash
$ pip install -y opencv-python==3.4.2.17
```
There are a few different options for installing Tensorflow on your system, which vary depending on the hardware you have, and the speedups that you want to include. You can install either the tensorflow or tensorflow-gpu packages from pip, or you can build them from source by [following the directions here](https://www.tensorflow.org/install/source).

N.B. Building and installing tensorflow from source takes a _long time_. However, it does offer a considerable speed boost (for me, it sped up by about 3x). If you can't find a pre-built wheel, like [the ones in this wonderful repository](https://github.com/lakshayg/tensorflow-build), then my recommendation is not to take this route. 

To install tensorflow with pip:
``` bash
$ pip install tensorflow==1.12
```
Or if you want to use the GPU to speed up processing:
``` bash
$ pip install tensorflow-gpu==1.12
```

As in the headings, you can follow the same steps on MacOS, Linux and Windows, and you will get the same results.

# Notes
If you are on Windows - especially versions before Windows 10, installation from the pip wheel may not work for you. Instead, try the instructions [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_setup/py_setup_in_windows/py_setup_in_windows.html).

There are a number of prebuilt tensorflow wheels that are posted publicly - you may be able to find these for your specific platform. I've included some links to such repos below:
* [For MacOS, Windows and Linux](https://github.com/lakshayg/tensorflow-build) (This is the same link as above)