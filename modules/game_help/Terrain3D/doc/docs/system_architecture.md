System Architecture
=====================

## Geometry Clipmap Terrain
Some terrain systems generate a grid of mesh chunks, centered around the camera. As the camera moves forward, old meshes far behind are destroyed and new meshes in front are created.

Like The Witcher 3, this system uses a geometry clipmap, where the mesh components are generated once, and at periodic intervals are centered on the camera location. On each update, the vertex heights of the mesh components are adjusted by the GPU in the vertex shader, by reading from the terrain heightmap. Levels of Detail (LODs) are built into the mesh generated on startup, so don't require any additional consideration once placed. Lower detail levels are automatically placed far away once all mesh components are recentered on the camera. See Mike Savage's blog for visual examples.

### Reference Material
* Mike J. Savage: [Geometry clipmaps: simple terrain rendering with level of detail](https://mikejsavage.co.uk/blog/geometry-clipmaps.html)

* NVidia GPU Gems 2: [Terrain Rendering Using GPU-Based Geometry Clipmaps](https://developer.nvidia.com/gpugems/gpugems2/part-i-geometric-complexity/chapter-2-terrain-rendering-using-gpu-based-geometry)

* GDC 2014: [The Witcher 3 Clipmap Terrain and Texturing](https://archive.org/details/GDC2014Gollent) - [Slides](https://ubm-twvideo01.s3.amazonaws.com/o1/vault/GDC2014/Presentations/Gollent_Marcin_Landscape_Creation_and.pdf)

## Architecture
Here is a diagram showing what the classes do and how they communicate.

```{image} images/sa_uml.png
:target: ../_images/sa_uml.png
```