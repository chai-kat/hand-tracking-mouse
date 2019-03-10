# Usage
Once you've got everything installed, the simplest way to run the software is to double click on the *ht_mouse.py* file that should appear in the downloaded folder. 

You should not need to set anything up and you should see the mouse moving as soon as the program starts running (usually within 1-2 seconds). In its current iteration, if you hover over a position for around 1 second then you will get a click.

In short:
* Run the program, when you see the camera turn on (most computers have a light beside the camera which turns on when the camera is running)
* Move your hand in front of the camera (and make sure the camera can see most of it)
* Hover over a point for about a second to click on it

There are some issues with using this software at the edges of the screen or very far away. Future fixes will take care of the second by updating the dataset used for hand recognition. If you want to use the edges of the screen better, you can use a wide-angle lens on your camera (think: [this](https://www.thingiverse.com/thing:478671)), or, even better, a native wide-angle camera. In case you're feeling a little less motivated, moving your a little further away does the trick pretty well. 

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

On MacOS and Linux you can use the provided installation script:
```bash
$ ./installLinux.sh
```
You can also double-click the installLinux file and it will handle the rest for you.
If you are on Windows, then skip to the directions for installing with pip.

Also, If you have conda installed to a directory that's not the default, don't want to use conda, or want to use a GPU-compatible/specially built TensorFlow package, read on.


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

N.B. Building and installing tensorflow from source takes a _long time_. However, it does offer a considerable speed boost (for me, it sped up by about 3x). If you can't find a pre-built wheel, like the ones in the notes section below, then my recommendation is not to take this route. Also, if you are using this, make sure to choose a wheel that was built for Python 2.7.15 and is built for TF Version 1.1.2. **The software may not work otherwise.** 

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

For this project, you **must** use *Tensorflow 1.12* and *Python 2.7.15*

If you use this I would much appreciate if you could fill out the following surveys:
* [Installation Experience Survey](https://docs.google.com/forms/d/1kLv4SNQ-jc7B72dzxahmLuUo6W9kVNQT3_dNKCX1CxA/edit)
* [Usage Experience Survey](https://docs.google.com/forms/d/1v_pfLsXHCjcyn0H4W6U2zS_usfi7vh-ugRUPmMv68JY/edit)
* Any other feedback is appreciated in the Issues page