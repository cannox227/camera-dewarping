# camera-dewarping
Camera dewarping for top-view court stitching

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
