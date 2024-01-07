# Project Status

Terrain3D has been public since July 2023 and is approaching Beta. We have hundreds of users and are using it in our games. It is stable, it's just missing advanced features.

Status of various features are shown in the table below.

See the [Roadmap](https://github.com/users/TokisanGames/projects/3/views/1) for priorities. See [Contributing](../contributing.rst) if you would like to help the project move faster.

| Feature | Status | 
| ------------- | ------------- | 
| **Platforms** | Terrain editing and exported games work on Windows, Linux, macOS. Experimental in 4.2: [Steam Deck](https://github.com/TokisanGames/Terrain3D/issues/220#issuecomment-1837552459), [Android](https://github.com/TokisanGames/Terrain3D/issues/197#issuecomment-1815513064). Pending: [IOS](https://github.com/TokisanGames/Terrain3D/pull/219), [WebGL](https://github.com/TokisanGames/Terrain3D/issues/217)
| **Languages** | GDScript, C++, C# all work, as should any language Godot supports. See [Integrating With Terrain3D](integrating.md)
| **Editing** |
| Sculpting Operations | Raise, Lower, Flatten, Expand (Multiply away from 0), Reduce (Divide towards 0), Smooth. Needs refinement
| Painting Operations | Texture, Color, Wetness (roughness) with Height blending. Needs work
| GPU Sculpting| [Pending](https://github.com/TokisanGames/Terrain3D/issues/174). Currently painting occurs on the CPU in C++. It's reasonably fast, but we limit the brush size to 200 as larger lags too much.
| Advanced texturing| [Pending](https://github.com/TokisanGames/Terrain3D/discussions/64) and [this](https://github.com/TokisanGames/Terrain3D/discussions/4). eg. Paintable uv scale / slope / rotation, 2-layer texture blending, 3D projection. We intend to implement all of these and adopt techniques provided by The Witcher 3 team. (See [System Design](system_architecture.md))
| **Environment** |
| Foliage | [Pending](https://github.com/TokisanGames/Terrain3D/issues/43). Non-collision based, paintable meshes (rocks, grass) will likely be added as a particle shader. Alternatives are [Scatter](https://github.com/HungryProton/scatter) once he has his particle shader working, your own particle shader, [Simple Grass Painter](https://godotengine.org/asset-library/asset/1623) (add as a child and enable debug collision) or [MultiMeshInstance3D](https://docs.godotengine.org/en/stable/tutorials/3d/using_multi_mesh_instance.html)
| Object placement | [Out of scope](https://github.com/TokisanGames/Terrain3D/issues/47). See 3rd party tools below.
| Holes | Supported in 0.9. See [#60](https://github.com/TokisanGames/Terrain3D/issues/60#issuecomment-1817623935)
| Water | Use [WaterWays](https://github.com/Arnklit/Waterways) for rivers, or [Realistic Water Shader](https://github.com/godot-extended-libraries/godot-realistic-water/) or [Infinite Ocean](https://stayathomedev.com/tutorials/making-an-infinite-ocean-in-godot-4/) for lakes or oceans.
| Destructibility | Real-time modification is technically possible by fetching the height and control maps and directly modifying them. That's how the editor works. But most gamedevs who want destructible terrains are better served by [Zylann's Voxel Tools](https://github.com/Zylann/godot_voxel).
| Non-destructive layers | Used for things like river beds, roads or paths that follow a curve and tweak the terrain. It's [possible](https://github.com/TokisanGames/Terrain3D/issues/129), but low priority.
| **Physics** |
| Godot | Works within regions you define in your world. No collision outside of those.
| Jolt | [Godot-Jolt](https://github.com/godot-jolt/godot-jolt) v0.6+ works as a drop-in replacement for Godot Physics. The above restriction applies.
| **Navigation Server** | Supported. See [Navigation](navigation.md)
| **Data** |
| Large terrains | 8k^2 works, maybe a little more, though [collision will take up ~3GB RAM](https://github.com/TokisanGames/Terrain3D/issues/161). 16k x 8k up to 16k^2 works in memory, but cannot be saved due to an [engine bug](https://github.com/TokisanGames/Terrain3D/issues/159).
| Importing / Exporting | Works. See [Importing data](import_export.md)
| Double precision floats | [Needs testing](https://github.com/TokisanGames/Terrain3D/issues/30)
| **Rendering** |
| Frustum Culling | The terrain is made up of several meshes, so half can be culled if the camera is near the ground.
| Occlusion Culling | Supported with our [custom baker](occlusion_culling.md).
| SDFGI | Works fine.
| Lightmap baking | Not possible. There is no static mesh, nor UV2 channel to bake lightmaps on to.
| **3rd Party Tools** |
| [Scatter](https://github.com/HungryProton/scatter) | For placing objects algorithmically, with or without collision. We provide [a script](https://github.com/TokisanGames/Terrain3D/blob/main/project/addons/terrain_3d/extras/project_on_terrain3d.gd) that allows Scatter to detect our terrain. 
| [AssetPlacer](https://cookiebadger.itch.io/assetplacer) | A level design tool for placing assets manually. Works on Terrain3D with `debug_show_collision` turned on. Native support is pending.
