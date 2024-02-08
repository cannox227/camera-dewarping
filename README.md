# Multi-sensor camera dewarping for basketball court view correction

This project involves the development of a Python application using the OpenCV library to reduce artifacts resulting from the mixing of multiple video streams from three multi-sensor cameras in a video recording system within a sports center gymnasium. By defining points of interest, users can establish triangles that can be manipulated to correct distortion caused by the recording system. The mapping between original and new positions is achieved through a technique known as dewarping, which effectively repairs most artifacts caused by object duplication and non-converging lines

|![](media/double-player.png)<br>Object duplication|![](media/line-cross.png)<br>Line not convering to the same point|
|:-:|:-:|

<div align="center">
  <video src="https://github.com/cannox227/camera-dewarping/assets/33253895/bebb5a97-0931-4b65-b69a-aefee43d86c2" style="width=50%" />
</div>

## Install and Run
### Requirements
- [Poetry](https://python-poetry.org/docs/basic-usage/) as package manager.
- [Python &ge; 3.10](https://www.python.org) 
- [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) to store and download the video/assets used

### Installation
0. Install manually the requirements
1. Clone the repo
```
git clone https://github.com/cannox227/camera-dewarping
```
2. Access the folder
```
cd camera-dewarping
```
3. Download assets (optional)
```
git lfs pull
```
4. Install required packages using poetry
```
poetry install
```
5. Activate Poetry shell
```
poetry shell
```
6. Run the script
- **Option 1:** run the script for the first time on a specific video source (to define the transformation)
  ```
  dewarp <filename>
  ```
- **Option 2:** run the script using a previously stored transformation matrix  (directly picked up from the output folder)
  ```
  dewarp <file-name> --load 
  ```
### Run the tool
Here below are reported all the possible flags and keybinding that can be used in the tool
```
Usage: dewarp [OPTIONS] FILE

  Dewarping tool for cameras
  Keybindings:
      - q: exit                               - s: save configuration
      - c: cancel selected point              - r: release point selection (deselect)
      - p: pop the last point                 - l: load saved data
      - g: create a new group of points       - v: toggle visibility of the warped area
      - Enter: advance to the next state      - d: toggle drawing of points and triangles

Options:
  --load         Load presaved config (from output folder)
  --scale FLOAT  Window scale wrt. video size
  --help         Show this message and exit.
```

<div style="text-align: center;">
    <img src="media/workflow-warping.png" alt="workflow-warping">
</div>

#### Triangle definition and warping
When interacting with the video source it is possible to define three or more points from which one or more triangle will be defined using the [Delaunay trianglulation](https://en.wikipedia.org/wiki/Delaunay_triangulation) technique. 
After having defined the triangles (with blue lines), by pressing `Enter` key we can start moving the triangles to the new positions.
In this phase we can move the triangleses (with gray lines), then when we are happy with the configuration we can press again `Enter` and effectively apply the warping.

|![](media/triangle_1.png)<br>Triangle definition|![](media/triangle_2.png)<br>Triangle can be moved to new position|
|:-:|:-:|

|![](media/triangle_3.png)<br>New position selected|![](media/triangle_4.png)<br>Warping applied|
|:-:|:-:|

<div style="display: flex; justify-content: center; align-items: center; height: 100vh;">
    <table>
        <tr>
            <td style="text-align: center;"><img src="media/triangle_5.png" alt="Final result"><br>Final result</td>
        </tr>
    </table>
</div>

<div align="center">
  <video src="https://github.com/cannox227/camera-dewarping/assets/33253895/386bcb30-5563-45fb-96de-b691753e3db9" style="width=50%" />
</div>

#### Handling different groups of triangles
Considering the fact that Delaunay triangulation returns a group of triangles that share the same points (some of which are indeed common vertices) and artifacts don't always occur in nearby areas of the court, it's necessary to ensure the user can define more than one group of triangles.

By pressing the `g` key, it's possible to define additional groups of triangles, thus applying more than one Delaunay triangulation completely independently.

<div align="center">
  <video src="https://github.com/cannox227/camera-dewarping/assets/33253895/03f500bd-53cf-4c51-8bcb-0a602d1dccfa" style="width=50%" />
</div>
    
#### Erasing a specific point / Erasing the last added point
Another proposed feature involves deleting a specific point by selecting it and pressing the `c` key (cancel), or deleting the last added point with the `p` key (pop).

<div align="center">
  <video src="https://github.com/cannox227/camera-dewarping/assets/33253895/203383c0-3a27-47ff-98a0-00ac9b8a21a5" style="width=50%" />
</div>


#### Saving the transformation
After having defined the new triangles and having applied the warping, it is possible to store the information regarding each group of old and new triangles and the correspective transformation matrix. This can be achieved by pressing the key `s`. The files are stored as Numpy array in the `output` folder in `.pkl` [format](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html)
