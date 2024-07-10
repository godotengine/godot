Installation & Upgrades
==========================

## Requirements
* Supports Godot 4.2+.
* Supports Windows, Linux, and macOS. Other platforms are [pending](project_status.md).

## Running the demo
1. Download the [latest binary release](https://github.com/TokisanGames/Terrain3D/releases) and extract the files, or [build the plugin from source](building_from_source.md).
2. Run Godot, using the console executable so you can see error messages.
3. In the project manager, import the demo project and open it. Allow Godot to restart.
4. In `Project Settings / Plugins`, ensure that Terrain3D is enabled.
5. Select `Project / Reload Current Project` to restart once more.
6. If the demo scene doesn't open automatically, open `demo/Demo.tscn`. You should see a terrain. Run the scene. 

If it isn't working for you, watch the [tutorial videos](tutorial_videos.md) and read [Troubleshooting](troubleshooting.md) and [Getting Help](getting_help.md).

## Installing Terrain3D in your project
1. Download the [latest binary release](https://github.com/TokisanGames/Terrain3D/releases) and extract the files, or [build the plugin from source](building_from_source.md).
2. Copy `addons/terrain_3d` to your project folder as `addons/terrain_3d`.
3. Run Godot, using the console executable so you can see error messages. Restart when it prompts.
4. In `Project Settings / Plugins`, ensure that Terrain3D is enabled.
5. Select `Project / Reload Current Project` to restart once more.
6. Create or open a 3D scene and add a new Terrain3D node.
7. Select Terrain3D in the Scene panel. In the Inspector, click the down arrow to the right of the `storage` resource and save it as a binary `.res` file. The other resources can be left as is or saved as text `.tres`. These external files can be shared with other scenes.
8. Click Next to learn how to properly [set up your textures](texture_prep.md), or skip to [import data](import_export.md).

If it isn't working for you, watch the [tutorial videos](tutorial_videos.md) and read [Troubleshooting](troubleshooting.md) and [Getting Help](getting_help.md).

## Upgrading Terrain3D

To update Terrain3D: 
1. Close Godot.
2. Remove `addons/terrain_3d` from your project folder.
3. Copy `addons/terrain_3d` from the new release download or from your build directory into your project folder.

Don't just copy the new folder over the old, as this won't remove any files that we may have intentionally removed.

## Upgrade Path

While later versions of Terrain3D can generally open previous versions, not all data will be loaded unless the supported upgrade path is followed. We occasionally deprecate or rename classes and provide upgrade paths to convert data for a limited time. 

Given the table below, to upgrade 0.8 to the latest version you would need to open your files in 0.8.4 or 0.9 and save them, then open in the latest version and save again.

| Starting Version | Supported Upgrade |
|------------------|-------------------|
| 0.8.4 - 0.9.0 | Latest |
| 0.8.0 - 0.8.3 | 0.8.4 - 0.9.0 |

