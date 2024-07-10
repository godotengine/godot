# Navigation

Navigation in games is quite a broad and sometimes complex topic. A full description can't be provided here, so it's recommended to familiarize yourself with the basic concepts in the [official Godot documentation](https://docs.godotengine.org/en/stable/tutorials/navigation/navigation_introduction_3d.html) first.

This page describes how to bake a nav mesh (navigation mesh) for your terrain.


## Setting Up Navigation

Nav meshes take a long time to bake, and in most games, it would be wasteful to generate vast amounts of navigation data for areas that navigation agents aren't going to use. So by default, all terrain is un-navigable. To make parts of it navigable, you must use the Navigable terrain tool. Navigable areas appear in dark magenta when this tool is selected.

```{image} images/nav_painting.png
:target: ../_images/nav_painting.png
```

Next, you will need a `NavigationRegion3D` node. If you don't already have one, Terrain3D provides a convenient tool to set one up for you. Select your `Terrain3D` node, then in the `Terrain3D Tools` menu on top, click `Set up Navigation`.

```{image} images/terrain3d_tools.png
:target: ../_images/terrain3d_tools.png
```

The same steps can be performed manually if you prefer:

1. Create a `NavigationRegion3D` node.
2. Assign it a blank `NavigationMesh` resource. Review and adjust the settings on it if you need to.
3. If using the default source geometry mode, move the `Terrain3D` node to be a child of the new `NavigationRegion3D` node. Otherwise, if you selected one of the group-based modes, add the Terrain3D node to the group.


## Baking a Nav Mesh

Once navigation has been set up, baking and re-baking it is straight-forward:

1. Select the `Terrain3D` node.
2. In Terrain3D Tools, click `Bake NavMesh`. This can take a long time to complete.

Note that the standard `Bake NavMesh` button that `NavigationRegion3D` provides will not generate a nav mesh for Terrain3D (see [godot-proposals#5138](https://github.com/godotengine/godot-proposals/issues/5138)). Only use the Terrain3D Tools baker, which appears whenever you click the `Terrain3D` node or any `NavigationRegion3D` nodes.

```{image} images/nav_baking.png
:target: ../_images/nav_baking.png
```

If this is your first time setting up and baking a nav mesh, the only thing left to do is add your navigation agents. See [Godot's very clear and thorough documentation on navigation agents](https://docs.godotengine.org/en/stable/tutorials/navigation/navigation_using_navigationagents.html), which provides several handy template scripts you can use.

You can also play with the NavigationDemo.tscn and CodeGenerated.tscn scenes which both demonstrate navigation.

<figure class="video_container">
 <video width="600px" controls="true" allowfullscreen="true">
 <source src="../_static/video/nav_demo.mp4" type="video/mp4">
 </video>
</figure>


## Tips

### Enable visible navigation for debugging

This option enables a blue overlay mesh that displays where the navigation mesh exists.

```{image} images/nav_debugging.png
:target: ../_images/nav_debugging.png
```

### Save NavigationMesh resources to disk

NavigationMesh resources can bloat the size of your scene. It's recommended to save these resources to disk in binary format with the `.res` extension.

### Use multiple nav meshes in large scenes

As mentioned, in many games, large areas of terrain are generally unreachable to agents. The 'Navigable Areas' tool is used to reduce the amount of geometry coming from the `Terrain3D` node, however having lots of other meshes in unreachable areas can also lead to long bake times.

If you have a very large scene in, for example, an open world RPG, it's better to have multiple small nav meshes that cover only what you need, rather than one giant one covering the entire world. In said example, each RPG town could have its own nav mesh. To do this, you would need to:

1. Create a NavigationRegion3D node for each town, each with their own NavigationMesh resources (i.e. unique, not shared).
2. Define the [`filter_baking_aabb`](https://docs.godotengine.org/en/stable/classes/class_navigationmesh.html#class-navigationmesh-property-filter-baking-aabb) on each nav mesh, so that it only bakes objects within its own area.
3. To use the same Terrain3D node with multiple NavigationRegion3D, set up the nav meshes to use one of the [`SOURCE_GEOMETRY_GROUPS_*` modes](https://docs.godotengine.org/en/stable/classes/class_navigationmesh.html#class-navigationmesh-property-geometry-source-geometry-mode) instead of the default `SOURCE_GEOMETRY_ROOT_NODE_CHILDREN`, and add the Terrain3D node to the group.


## Common Issues

### NavigationMeshSourceGeometryData3D is empty. Parse source geometry first.

The engine produces this error if there's nothing for a NavigationRegion3D to generate a nav mesh from. The most likely cause, if you're using Terrain3D, is that you haven't painted any parts of the terrain as navigable.

### Navigation map synchronization error

`Navigation map synchronization error. Attempted to merge a navigation mesh polygon edge with another already-merged edge. This is usually caused by crossing edges, overlapping polygons, or a mismatch of the NavigationMesh / NavigationPolygon baked 'cell_size' and navigation map 'cell_size'`

There are several possible causes for this. If the `cell_size` of your nav mesh matches the `cell_size` in your project settings, it's currently believed to be caused by [an engine bug](https://github.com/godotengine/godot/issues/85548). This error message shouldn't affect the usability of your nav meshes.

### Agents get stuck on collisions, run in circles, go off the nav mesh, or fail to find obvious paths

Developing good path-following behaviors is a very complex topic, far beyond the scope of this article. In general, make sure your NavigationMesh settings, NavigationAgent3D settings, and collisions are all consistent with each other. If a NavigationAgent3D is using a NavigationMesh that was baked for smaller agents than itself, for instance, then it's going to get stuck.

You can try repainting an area and regenerating the nav mesh.

Making reasonable fallback behaviors, when you're able to detect in a script that something has gone awry, can also help.


## Baking a Nav Mesh at Runtime

If your project has dynamic, or generated terrain, or if the traversable area of your terrain is so gigantic that it can't be baked in the editor, then you might need to use runtime navmesh baking.

Terrain3D contains an example script that shows how to bake terrain nav meshes at runtime, which you can find in the `CodeGenerated.tscn` demo scene. 

<figure class="video_container">
 <video width="600px" controls="true" allowfullscreen="true">
 <source src="../_static/video/nav_code_demo.mp4" type="video/mp4">
 </video>
</figure>

The script periodically re-bakes a nav mesh in the area around the player as it moves through the scene. Adjust `bake_cooldown` if you wish the baker to update more frequently, at the cost of CPU cycles. 

You can also adjust how frequently the navigation agent updates its path by adjusting `Enemy.gd:RETARGET_COOLDOWN`. If you have a lot of agents, that's going to come with a performance hit.

### Performance Tips

Navigation baking is slow. Editor or compile-time baking navigation is usually a better option for games. Runtime baking can still be usable however. There are a few things you can do to speed it up, or work around the slowness:

* Create a set of fallback behaviors that get used when proper navigation isn't possible. In this way, you can make the AI degrade without completely failing while waiting for the nav server, or when out of range of the nav mesh. For instance the enemy could simply move straight towards the player if navigation is not ready yet. It could have rudimentary ability to detect cliffs and obstacles with raycasts.
* Reduce the speed of the player character. Delays happen frequently in the demo because the player can move across the mesh so rapidly.
* Reduce the size of the baked mesh with `mesh_size` to make it cheaper to bake.
* Increase the size of the cells (i.e. reduce the resolution of the navmesh) to reduce the amount of work to do. Change `cell_size` in the `template` NavigationMesh. If you increase it too far, obstacles may hide within a cell and break your navigation. You can write fallback behaviors for that as well, or ensure that your obstacles are all larger than the cell size.
