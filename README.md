# Subdivide

![Example, created from https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Volc%C3%A1n_Chimborazo%2C_%22El_Taita_Chimborazo%22.jpg/640px-Volc%C3%A1n_Chimborazo%2C_%22El_Taita_Chimborazo%22.jpg](./mountain.jpg_iteration535.png)

A python script to create colored triangular meshes from images.

## Installation
You'll need to install OpenCV with python bindings. I.e.
``` sh
sudo apt-get install python-opencv
```

Additionally, install the python packages for click and numpy.

## Usage

``` sh
./subdivide.py --until-distance=900 --interactive <some-image-file>
```

