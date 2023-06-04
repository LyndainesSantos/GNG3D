# GNG3D
This is a part of my master's degree research using Growing Neural Gas (GNG) and 3D data image from Kinect device. The goal of this repository is processing pcd format files into 3D data corresponding to edges and vortex. 

## Introduction

Growing Neural Gas (GNG) is a widely used algorithm in mobile robotics to enhance the learning capability of autonomous systems. GNG is a neural network that has the ability to grow and adapt to the environment, allowing mobile robots to acquire knowledge and adjust to new situations. This algorithm is particularly useful for creating cognitive maps, where the robot can map and understand the environment in real-time, identifying obstacles, safe areas, and other relevant information for autonomous navigation. 

By utilizing GNG, mobile robots can :

- improve their ability to make intelligent decisions,
- optimize routes,
- avoid collisions,
- efficiently explore unknown environments and so on. 

Thus, Growing Neural Gas plays a crucial role in mobile robotics, providing a solid foundation for enhancing the intelligence and autonomy of robots in their interactions with the physical world.

## Project's Organization

This repository is organized according to the table below:

| Directory / File  | Description |
|-------|-------|
| ./database  | directory for all files used to trained and test the neural network's performance    |
| ./image_io  |  there is the auxiliary functions to load, process, compute and for visualization tasks|
| ./result    | edges and vortex 3D data JSON files |
| gng3d_reconstruction.py  |  GNG processing   |
| main.py     | script to set paths and calls the main funtions for processing the tasks |

## To Install
This project was developed on Ubuntu [22.04](https://ubuntu.com/download/desktop) and [Python 3.7](https://www.python.org/downloads/release/python-370/) operating system.
For install all the dependencies you may run:

```bash
pip install -r requirements.txt
```
This will install the necessaries libraries used to performe the GNG at 3D points, such as:
- [open3D](http://www.open3d.org/)
- [bagpy](https://pypi.org/project/bagpy/)
- [graphviz](https://pypi.org/project/graphviz/)
- [keras](https://keras.io/)
- [opencv](https://docs.opencv.org/3.4/index.html)
- [pandas](https://pandas.pydata.org/)
- [Pillow](https://pypi.org/project/Pillow/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/)
- [tensorflow](https://www.tensorflow.org/?hl=pt-br)

## To Run

To compute the GNG and to show the 3D graph, just run:

```bash
python main.py
```
or

```bash
python3 main.py
```
This script will generate GNG through 3D plot according to the input data.

Some paths are as default in main.py, such as:
- Path for read the Point Cloud data file: path_single_pcd = "./database/bag2pcdRGB"

- Path to save the edges and vortex data files: path_es_vs_result = "./result"

## Results

Here are some brief results by performing the GNG algorithm. 

The input data, aquired through Kinect device:

https://github.com/LyndainesSantos/GNG3D/assets/125848451/61fff536-0658-4312-9d4f-94a343847aa4

In the figure below it is possible to see the comparison of the image on the left, which represents the point cloud input. And on the right, we have the resulting GNG from that input.

![scene_1_compare.png](images_videos%2Fscene_1_compare.png)
