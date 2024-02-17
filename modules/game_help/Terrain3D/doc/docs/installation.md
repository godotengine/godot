Installation
==============

## Requirements
* Supports Godot 4.1+. [4.0 is possible](previous_engines.md).
* Supports Windows, Linux, and macOS. Other platforms are [pending](project_status.md).

## Run the demo
1. Download the [latest release](https://github.com/TokisanGames/Terrain3D/releases) and extract the files, or [build the plugin from source](building_from_source.md).
2. Run Godot, using the console executable so you can see error messages.
3. In the project manager, import the demo project and open it. Allow Godot to restart.
4. In `Project Settings / Plugins`, ensure that Terrain3D is enabled.
5. Select `Project / Reload Current Project` to restart once more.
6. If the demo scene doesn't open automatically, open `demo/Demo.tscn`. You should see a terrain. Run the scene. 

If it isn't working for you, watch the [tutorial videos](tutorial_videos.md) and read [Troubleshooting](troubleshooting.md) and [Getting Help](getting_help.md).

## Install Terrain3D in your own project
1. Download the [latest release](https://github.com/TokisanGames/Terrain3D/releases) and extract the files, or [build the plugin from source](building_from_source.md).
2. Copy `addons/terrain_3d` to your project folder as `addons/terrain_3d`.
3. Run Godot, using the console executable so you can see error messages. Restart when it prompts.
4. In `Project Settings / Plugins`, ensure that Terrain3D is enabled.
5. Select `Project / Reload Current Project` to restart once more.
6. Create or open a 3D scene and add a new Terrain3D node.
7. Select Terrain3D in the Scene panel. In the Inspector, click the down arrow to the right of the `storage` resource and save it as a binary `.res` file. The other resources can be left as is or saved as text `.tres`. These external files can be shared with other scenes.
8. Click Next to learn how to properly [set up your textures](texture_prep.md), or skip to [import data](import_export.md).

If it isn't working for you, watch the [tutorial videos](tutorial_videos.md) and read [Troubleshooting](troubleshooting.md) and [Getting Help](getting_help.md).

