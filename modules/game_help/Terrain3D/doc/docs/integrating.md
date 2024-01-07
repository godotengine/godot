Integrating with Terrain3D
===========================

This page is for more advanced gamedevs who want to access Terrain3D from other languages like C# or those developing tools & addons.

Any language Godot supports should work via the GDExtension interface.

```{image} images/integrating_gdextension.png
:target: ../_images/integrating_gdextension.png
```

## Detecting If Terrain3D Is Installed

To determine if Terrain3D is installed and active, [ask Godot](https://docs.godotengine.org/en/stable/classes/class_editorinterface.html#class-editorinterface-method-is-plugin-enabled).

**GDScript**
```gdscript
#4.2
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

## Detecting the Terrain3D Node

* If collision is enabled in game (default) or in the editor (debug only), you can run a raycast and if it hits, it will return a `Terrain3D` object. See more below in the raycast section.

* Your script can provide a NodePath and allow the user to select their Terrain3D node as was done in [the script](https://github.com/TokisanGames/Terrain3D/blob/df901b4fd51a81175e4f5177c33318a8a4b19c36/project/addons/terrain_3d/extras/project_on_terrain3d.gd#L13) provided for use with Scatter.

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

## Calling Terrain3D

**GDScript**

Terrain3D is treated like any other object. Just make a new object.

```gdscript
     var terrain: Terrain3D = Terrain3D.new()
     terrain.storage = Terrain3DStorage.new()
     terrain.texture_list = Terrain3DTextureList.new()
     print(terrain.storage.get_version())
```

**C#**

You can instantiate through ClassDB, set variables and call it.

```c#
     var terrain = ClassDB.Instantiate("Terrain3D");
     terrain.AsGodotObject().Set("storage", ClassDB.Instantiate("Terrain3DStorage"));
     terrain.AsGodotObject().Set("texture_list", ClassDB.Instantiate("Terrain3DTextureList"));
     terrain.AsGodotObject().Call("set_collision_enabled", true);
```

You can also check if a node is a Terrain3D object:

```c#
private bool CheckTerrain3D(Node myNode) {
        if (myNode.IsClass("Terrain3D")) {
            var debugCollisions = myNode.Call("get_show_debug_collision").AsInt32();
```

For more information read [Cross-language scripting](https://docs.godotengine.org/en/stable/tutorials/scripting/cross_language_scripting.html) in the Godot docs.


## Detecting Terrain Height

There are two ways to determine the height in Terrain3D.

### 1) Raycasting

Normally, collision is not generated in the editor. If `Terrain3D.debug_show_collision` is enabled, it will generate collision in the editor and you can do a normal raycast. This mode also works fine while running in a game.

This debug option will generate collision one time when enabled or at startup. If the terrain is sculpted afterwards, this collision will be inaccurate to the visual mesh until it is disabled and enabled again. On a Core-i9 12900H, generating collision takes about 145ms per region, so updating it several times per second while sculpting is not practical. Currently all regions are regenerated, rather than only modified regions so it is not optimal.

There is no collision outside of regions in any mode. See below.


### 2) Querying

The much more optimal way to detect height is to just ask the Terrain3DStorage for the height directly:

```gdscript
     var height: float = terrain.storage.get_height(global_position)
```

This is ideal for one lookup. However, if you wish to look up thousands of heights, it will be significantly faster if you retrieve the heightmap Image for the region and query it directly.

```gdscript
     var region_index: int = terrain.storage.get_region_index(global_position)
     var img: Image = terrain.storage.get_map_region(Terrain3DStorage.TYPE_HEIGHT, region_index)
     for y in img.get_height():
          fox x in img.get_width():
               var height: float = img.get_pixel(x, y).r
```


### Regions

Terrain3D provides users non-contiguous 1024x1024 sized regions on a 16x16 region grid. So a user might have a 1k x 2k island in one corner of the world and a 4k x 4k island elsewhere. In between are empty regions, visually flat space where they could place an ocean. No collision is generated there. 

If you run a raycast outside of the regions, it won't hit anything.

If you query `get_height()` outside of regions, it will return 0.

You can determine if a given location is within a region by using this function. It will return -1 if the XZ location is not within a region. Y is ignored.

```gdscript
     var region_index: int = Terrain3DStorage.get_region_index(global_position)
```
