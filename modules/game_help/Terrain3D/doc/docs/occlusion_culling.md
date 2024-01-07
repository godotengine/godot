Occlusion Culling
===================

Occlusion culling allows the renderer to hide objects that are behind the terrain. See the example below. For more information about occlusion culling in Godot, see [the official docs](https://docs.godotengine.org/en/stable/tutorials/3d/occlusion_culling.html).

<figure class="video_container">
 <video width="600px" controls="true" allowfullscreen="true">
 <source src="../_static/video/oc_demo.mp4" type="video/mp4">
 </video>
</figure>

## Baking Terrain Occlusion

First, enable `use_occlusion_culling` in the project settings. 
Then in the editor:

* Select Terrain3D.
* Click `Terrain3D Tools`, then `Bake Occluder3D` in the top menu. 
* On the popup window accept the default, LOD 4.

```{image} images/terrain3d_tools.png
:target: ../_images/terrain3d_tools.png
```

* Select the OccluderInstance3D child node.
* In the inspector, click the arrow to the right of the ArrayOccluder resource and choose save. Save the file as a binary `.occ`.

```{image} images/oc_save.png
:target: ../_images/oc_save.png
```

### More Information
The LOD value determines the granularity of the occlusion mesh, and therefore the number of vertices used. Baking an occluder at a lower level of detail (higher number) will reduce opportunities for culling, but make occlusion testing quicker.

Baking pauses the editor for about 5 seconds per region at LOD4. It has to read every pixel on the height map once to make sure that the generated occluder doesn't extend above or outside the clipmap (at any level of detail).

After baking completes, an OccluderInstance3D node is created as a child of the Terrain3D node with an Occluder3D resource in it containing the baked occlusion mesh.

The generated Occluder3D resource can be quite large. It's more efficient to store this in binary format than text format, so you should always save this resource to a `.occ` file after it has been baked.

The occluder has to be manually baked again each time the terrain is altered.

## Baking Occlusion For An Entire Scene

Godot has a built-in tool for baking occlusion for all meshes. It is visible when you add an OccluderInstance3D to the scene tree and select it.

```{image} images/oc_oc_menu.png
:target: ../_images/oc_oc_menu.png
```

This tool doesn't know about Terrain3D, so it will bake all MeshInstances and ignore Terrain3D. To get a complete bake, you will need to use our tool to bake the terrain occluder, then add a separate OccluderInstance3D to your scene and bake all of your other meshes.
