Double Precision
=================

Building Terrain3D with double precision (aka 64-bit) floats is possible, but there are some caveats.

When the player and camera move 10s-100s of thousands, or millions of units away from the origin using a single precision float engine, things start to break down. Movements, positions, and rendering starts to become jittery or corrupted as the gradation between values gets larger and larger.

Building with double precision floats allows the step between values to remains very small even at large numbers.

For a more detailed explanation, see [Large World Coordinates](https://docs.godotengine.org/en/stable/tutorials/physics/large_world_coordinates.html) in the Godot documentation.


## Caveats

* This feature is experimental and has had only one user give a positive report so far.
* There are many caveats listed in the link above. You should read them all before beginning this process.
* You must build Godot and Terrain3D from source.
* Terrain3D currently only supports a maximum world of 16k by 16k, or (+/-8192, +/-8192). Although with `mesh_vertex_spacing`, you can expand this up to 10x. You can also have Terrain3D located around the origin, then have your own meshes or a shader generated terrain outside of that world. See [Support more region sizes #70](https://github.com/TokisanGames/Terrain3D/issues/77) for supporting worlds up to 90k per side and more.
* Shaders do not support double precision. Clayjohn wrote an article demonstrating how to [Emulate Double Precision](https://godotengine.org/article/emulating-double-precision-gpu-render-large-worlds/) in shaders. He wrote that the camera and model transform matrices needed to be emulated to support double precision. This is now done automatically in the engine when building it with double precision. There may be other cases where shaders will need this emulation.


## Setup

One user reported success using Godot 4.2.1 and Terrain3D 0.9.1.

To get Terrain3D and Godot working with double precision floats, you must:

1. [Build Godot from source](https://docs.godotengine.org/en/latest/contributing/development/compiling/index.html) with `scons precision=double` along with your other parameters like target, platform, etc.
2. [Generate custom godot-cpp bindings](https://docs.godotengine.org/en/latest/tutorials/scripting/gdextension/gdextension_cpp_example.html#building-the-c-bindings) using this new executable. This will generate a JSON file you will use in the next step.
3. [Build Terrain3D from source](building_from_source.md) (which includes godot-cpp) using `scons precision=double custom_api_file=YOUR_CUSTOM_JSON_FILE`

After that, you can run the double version of Godot with the double version of Terrain3D.


## Further Reading

See [Support Doubles #30](https://github.com/TokisanGames/Terrain3D/issues/30) for our Issue that documents implementing support for doubles in Terrain3D.