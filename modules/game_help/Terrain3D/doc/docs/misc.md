Miscelaneous Features
=======================

This page collects various features that haven't been fully detailed or made their way into specific pages yet.


## Keyboard & Mouse Operations
Aka keyboard shortcuts, Hotkeys or commands (for searching)

### Asset Dock
* Left-click to select
* Middle-click to clear, removal is only possible at the end of the list
* Right-click to edit

### Foliage instancing
* Hold `CTRL` while painting to remove


### Future, Sculpting
* Hold CTRL to do the opposite operation
* Hold SHIFT to use the smooth brush


## Terrain3DObjects

This node allows you to snap objects to the terrain when sculpting or moving.

Objects that are children of this node will maintain the same vertical offset relative to the terrain as they are moved laterally. 

For example you can place a sphere on the ground, move it laterally where a hill exists, and it will snap up to the ground. Or you can lower the ground and the sphere will drop with the ground changes. 

You can then adjust the vertical position of the sphere so it is half embedded in the ground, then repeat either of the above and the sphere will snap with the same vertical offset, half embedded in the ground.

To use it:
* Add a new node
* Find Terrain3DObjects
* Add nodes as children of this node



## Minimal Shader

You'll find a shader that provides just the minimum needed to allow the terrain to work in `extras/minimum.gdshader`.

It includes no texturing. 

Load this shader into the override shader slot and enable it. Customize as you see fit.
