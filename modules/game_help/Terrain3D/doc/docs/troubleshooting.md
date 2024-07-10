Troubleshooting
=================

## Tutorial Videos

Make sure to watch the [tutorial videos](tutorial_videos.md) which show proper installation and setup.


## Use the Console

As a gamedev, you should always be running with the console open. This means you ran `Godot_v4.*_console.exe` or ran Godot in a terminal window.

```{image} images/console_exec.png
:target: ../_images/console_exec.png
```

This is what it looks like this when you have the console open. 

```{image} images/console_window.png
:target: ../_images/console_window.png
```

Godot, Terrain3D, and every other addon gives you additional information here, and when things don't work properly, these messages often tell you exactly why.

Godot also has an `Output` panel at the bottom of the screen, but it is slow, will skip messages if it's busy, and not all messages appear there.


## Debug Logs

Terrain3D has debug logs for everything, which it can dump to the [console](#use-the-console). These logs *may* also be saved to Godot's log files on disk.

Set `Terrain3D.debug_level` to `Info` or `Debug` and you'll get copious activity logs that will help troubleshoot problems.

You can also enable debugging from the command line by running Godot with `--terrain3d-debug=<LEVEL>` where `<LEVEL>` is one of `ERROR`, `INFO`, `DEBUG`, `DEBUG_CONT`. Debug Continuous (DEBUG_CONT) is for repetitive messages such as those that appear on camera movement.

To run the demo from the command line with debugging, open a terminal, and change to the project folder (where `project.godot` is):

Adjust the file paths to your system. The console executable is not needed since you're already running these commands in a terminal window.

```
# 1. Change to the project folder
cd <PATH TO PROJECT FOLDER>

# 2a. Run the project with debugging enabled at startup
<PATH TO GODOT EXECUTABLE> --terrain3d-debug=DEBUG

# 2b. Or run the editor with debugging enabled at startup
<PATH TO GODOT EXECUTABLE> -e --terrain3d-debug=DEBUG


# E.g. To run the demo with debug messages
cd /c/gd/Terrain3D/project
/c/gd/bin/Godot_v4.1.3-stable_win64.exe --terrain3d-debug=DEBUG

# Load the editor with debug messages
/c/gd/bin/Godot_v4.1.3-stable_win64.exe -e --terrain3d-debug=DEBUG
```

When asking for help on anything you can't solve yourself, you'll need to provide a full log from your console or file system. As long as Godot doesn't crash, it saves the log files on your drive. In Godot select, `Editor, Open Editor Data/Settings Menu`. On windows this opens `%appdata%\Godot` (e.g. `C:\Users\%username%\AppData\Roaming\Godot`). Look under `app_userdata\<your_project_name>\logs`. Otherwise, you can copy and paste messages from the console window above.

## Common Issues

### Startup Errors

#### After clicking the Terrain3D node, there are no editor tools or texture panel.

Enable the plugin, as described in the [installation instructions](installation.md).


#### The editor tools panel is there, but the buttons are blank or missing.

Restart Godot twice before using, as described in the [installation instructions](installation.md).


#### Start up is very slow. Or the scene or storage file is huge.

Save the storage resource as a binary `.res` file, as described in the [installation instructions](installation.md). You've likely saved it as text `.tres`, or didn't save it separately at all, which means a ton of terrain data is saved as text in the scene.

Alternatively, you have a large terrain and are generating collision for all of it. Disable collision, or set it to dynamic to create only a small collision around the camera (pending implementation, see [PR #278](https://github.com/TokisanGames/Terrain3D/pull/278)).


#### Unable to load addon script from path

`Unable to load addon script from path: xxxxxxxxxxx. This might be due to a code error in that script. Disabling the addon at 'res://addons/terrain_3d/plugin.cfg' to prevent further errors."`

Most certainly you've installed the plugin improperly. These are the common causes:

1) You downloaded the repository code, not a [binary release](https://github.com/TokisanGames/Terrain3D/releases).

2) You moved the files into the wrong directory. The Terrain3D files should be in `project/addons/terrain_3d`. `Editor.gd` should be found at `res://addons/terrain_3d/editor/editor.gd`. [See an example issue here](https://github.com/TokisanGames/Terrain3D/issues/200).  

Basically, the required library isn't where it's supposed to be. The error messages will tell you exactly the file name and path it's looking for. View that location on your hard drive. On windows you might be looking for `libterrain.windows.debug.x86_64.dll`. Does that file exist where it's looking in the logs? Probably not. Download the correct package, and review the instructions to install the files in the right location.


### Crashing

#### Godot Crashes on Load

If this is the first startup after installing the plugin, this is normal due to a bug in the engine currently. Restart Godot.

If it still crashes, try the demo scene. 

If that doesn't work, most likely the library version does not match the engine version. If you downloaded a release binary, download the exactly matching engine version. If you built from source review the [instructions](building_from_source.md) to make sure your `godot-cpp` directory exactly matches the engine version you want to use. 

If the demo scene does work, you have an issue in your project. It could be a setting or file given to Terrain3D, or it could be anywhere else in your project. Divide and conquer. Copy your project and start ripping things out until you find the cause.

#### Exported Game Crashes On Startup

First make sure your game works running in the editor. Then ensure it works as a debug export with the console open. If there are challenges, you can enable [Terrain3D debugging](#debug-logs) before exporting with debug so you can see activity. Only then, test in release mode. 

Make sure you have both the debug and release binaries on your system, or have built Terrain3D in [both debug and release mode](building_from_source.md#5-build-the-extension), and that upon export both libraries are in the export directory (eg. `libterrain.windows.debug.x86_64.dll` and `libterrain.windows.release.x86_64.dll`). If you don't have the necessary libraries, your game will close instantly upon startup.

### Importing Textures

#### WARNING: Terrain3D::_texture_is_valid: Invalid format. Expected DXT5 RGBA8.

Read [Preparing Textures](texture_prep.md) to learn how to properly channel pack your textures.

#### Albedo/Normal textures do not have same size! and the terrain turns white

Read [Preparing Textures](texture_prep.md) and review your textures for odd sizes. All textures must be the same size. eg. If the first set is 2k, all other textures need to be 2k as well.

## Debug Collision

Collision is generated at runtime using the physics server. Normally these collision shapes are not visible. 

To see collision in the editor, enable `Terrain3D.debug_show_collision`, or in the inspector, Debug `show_collision`. It won't regenerate the collision when you edit, but you can disable and reenable this flag to regenerate.

To see collision in game, enable `Terrain3D.debug_show_collision`, and in the editor menu, enable `Debug/Visible Collision Shapes`.