Integrating with Terrain3D
===========================

This page is for more advanced gamedevs who want to access Terrain3D from other languages like C# or those developing tools & addons.

Any language Godot supports should be able to work with Terrain3D via the GDExtension interface.

```{image} images/integrating_gdextension.png
:target: ../_images/integrating_gdextension.png
```

## Detecting If Terrain3D Is Installed

To determine if Terrain3D is installed and active, [ask Godot](https://docs.godotengine.org/en/stable/classes/class_editorinterface.html#class-editorinterface-method-is-plugin-enabled).

**GDScript**
```gdscript
#4.2+
     print("Terrain3D installed: ", EditorInterface.is_plugin_enabled("terrain_3d"))

#pre-4.2
     var ei: EditorInterface = EditorScript.new().get_editor_interface()
     print("Terrain3D installed: ", ei.is_plugin_enabled("terrain_3d"))
```

**C#**
```c#
     GetEditorInterface().IsPluginEnabled("terrain_3d") 
```

You can also ask ClassDB if the class exists:

**GDScript**
```gdscript
     ClassDB.class_exists("Terrain3D")
     ClassDB.can_instantiate("Terrain3D")
```
**C#**
```c#
     ClassDB.ClassExists("Terrain3D");
     ClassDB.CanInstantiate("Terrain3D");
```

## Instantiating & Calling Terrain3D

Terrain3D is instantiated and referenced like any other object.

**GDScript**

```gdscript
     var terrain: Terrain3D = Terrain3D.new()
     terrain.storage = Terrain3DStorage.new()
     terrain.assets = Terrain3DAssets.new()
     print(terrain.get_version())
```

See the `CodeGenerated.tscn` demo for an example of initiating Terrain3D from script.

**C#**

You can instantiate through ClassDB, set variables and call it.

```c#
     var terrain = ClassDB.Instantiate("Terrain3D");
     terrain.AsGodotObject().Set("storage", ClassDB.Instantiate("Terrain3DStorage"));
     terrain.AsGodotObject().Set("assets", ClassDB.Instantiate("Terrain3DAssets"));
     terrain.AsGodotObject().Call("set_collision_enabled", true);
```

You can also check if a node is a Terrain3D object:

**GDScript**

```gdscript
    if node is Terrain3D:
```

**C#**

```c#
private bool CheckTerrain3D(Node myNode) {
        if (myNode.IsClass("Terrain3D")) {
            var debugCollisions = myNode.Call("get_show_debug_collision").AsInt32();
```

For more information on C# and other languages, read [Cross-language scripting](https://docs.godotengine.org/en/stable/tutorials/scripting/cross_language_scripting.html) in the Godot docs.

## Capturing the Terrain3D Instance

These options are for programming scenarios where a user action is intented to provide your code with the Terrain3D instance.

* If collision is enabled in game (default) or in the editor (debug only), you can run a raycast and if it hits, it will return a `Terrain3D` object. See more below in the [raycasting](#raycasting-with-physics) section.

* Your script can provide a NodePath and allow the user to select their Terrain3D node as was done in [the script](https://github.com/TokisanGames/Terrain3D/blob/v0.9.1-beta/project/addons/terrain_3d/extras/project_on_terrain3d.gd#L14) provided for use with Scatter.

* You can search the current scene tree for [nodes of type](https://docs.godotengine.org/en/stable/classes/class_node.html#class-node-method-find-children) "Terrain3D".
```gdscript
     var terrain: Terrain3D # or Node if you aren't sure if it's installed
     if Engine.is_editor_hint(): 
          # In editor
          terrain = get_tree().get_edited_scene_root().find_children("*", "Terrain3D")
     else:
          # In game
          terrain = get_tree().get_current_scene().find_children("*", "Terrain3D")

     if terrain:
          print("Found terrain")
```

## Understanding Regions

Terrain3D provides users non-contiguous 1024x1024 sized regions on a 16x16 region grid. So a user might have a 1k x 2k island in one corner of the world and a 4k x 4k island elsewhere. In between are empty regions, visually flat space where they could place an ocean. In these empty regions, no vram is consumed, nor collision generated.

Outside of regions, raycasts won't hit anything, and querying terrain intersections will return NANs or INF (i.e. >3.4e38).

You can determine if a given location is within a region by using `Terrain3DStorage.get_region_index(global_position)`. It will return -1 if the XZ location is not within a region. Y is ignored.


## Detecting Terrain Height or Position

There are multiple ways to detect an intersection with the terrain. After which you may wish to use `Terrain3DStorage.get_normal(global_position)`.

### Query the height at any position

You can ask Terrain3DStorage for the height at any given location:

```gdscript
     var height: float = terrain.storage.get_height(global_position)
```

This is ideal for one lookup. However, if you wish to look up thousands of heights, it might be faster to retrieve the heightmap Image for the region and query it directly. However, note that `get_height()` will interpolate between vertices, while this code will not. 

```gdscript
     var region_index: int = terrain.storage.get_region_index(global_position)
     var img: Image = terrain.storage.get_map_region(Terrain3DStorage.TYPE_HEIGHT, region_index)
     for y in img.get_height():
          for x in img.get_width():
               var height: float = img.get_pixel(x, y).r
```


### Raycasting with Physics

Normally, collision is not generated in the editor. If `Terrain3D.debug_show_collision` is enabled, it will generate collision in the editor and you can do a normal raycast. This mode also works fine while running in a game.

This debug option will generate collision one time when enabled or at startup. If the terrain is sculpted afterwards, this collision will be inaccurate to the visual mesh until it is disabled and enabled again. On a Core-i9 12900H, generating collision takes about 145ms per region, so updating it several times per second while sculpting is not practical. Currently all regions are regenerated, rather than only modified regions so it is not optimal.

There is no collision outside of regions, so raycasts won't hit.

See the Godot docs to learn how to use physics based [Ray-casting](https://docs.godotengine.org/en/stable/tutorials/physics/ray-casting.html).


### Raycasting without Physics

It is possible to cast a ray from any given position and detect the collision point on the terrain using the GPU instead of the physics engine.

Sending the source point and ray direction to [Terrain3D.get_intersection()](https://terrain3d.readthedocs.io/en/latest/api/class_terrain3d.html#class-terrain3d-method-get-intersection) will return the intersection point on success.

You can review [editor.gd](https://github.com/TokisanGames/Terrain3D/blob/v0.9.1-beta/project/addons/terrain_3d/editor/editor.gd#L129-L143) to see an example of projecting the mouse position onto the terrain using this function.


## Getting Updates on Terrain Changes

`Terrain3DStorage` has [signals](https://terrain3d.readthedocs.io/en/latest/api/class_terrain3dstorage.html#signals) that fire when updates occur. You can connect to them to receive updates.


