Building from Source
====================

If you wish to use more recent builds without building from source, see [Nightly Builds](nightly_builds.md).

## 1. Install dependencies

Follow Godot's instructions to set up your system to build Godot. You don't need the Godot source code, so stop before then. You only need the build tools, specifically `scons`, `python`, a compiler, and any other tools these pages identify. They provide easy installation instructions.

* [Windows](https://docs.godotengine.org/en/latest/contributing/development/compiling/compiling_for_windows.html)
* [Linux/BSD](https://docs.godotengine.org/en/latest/contributing/development/compiling/compiling_for_linuxbsd.html)
* [macOS](https://docs.godotengine.org/en/latest/contributing/development/compiling/compiling_for_macos.html)


## 2. Download this repository

You can either grab the zip file, or clone it on the command line. Only type in the commands after the $ prompts.

```
$ git clone git@github.com:TokisanGames/Terrain3D.git

Cloning into 'Terrain3D'...
Enter passphrase for key:
remote: Enumerating objects: 125, done.
remote: Counting objects: 100% (125/125), done.
remote: Compressing objects: 100% (79/79), done.
remote: Total 125 (delta 56), reused 94 (delta 36), pack-reused 0
Receiving objects: 100% (125/125), 42.20 KiB | 194.00 KiB/s, done.
Resolving deltas: 100% (56/56), done.

$ cd Terrain3D

Terrain3D$ git submodule init
Submodule 'godot-cpp' (https://github.com/godotengine/godot-cpp.git) registered for path 'godot-cpp'

Terrain3D$ git submodule update
Cloning into 'C:/GD/Terrain3D/godot-cpp'...
Submodule path 'godot-cpp': checked out '9d1c396c54fc3bdfcc7da4f3abcb52b14f6cce8f'

```

Note the version it checked out: **9d1c396**...

This hash number is important for the next section.


## 3. Identify the appropriate godot-cpp version

The checked out version of the godot-cpp submodule needs to match the version of your Godot engine build. e.g. Godot Engine 4.0.2 official build with godot-cpp checked out to a 4.0.2 branch. The early days of Godot 4.x were very strict and required the exact same major, minor, and patch versions. Since then, the requirements have loosened. For instance we've matched godot-cpp 4.1.3 with Godot engine 4.1.3 through 4.2.1 without issue.

In the repository, we leave godot-cpp linked to an older version for broad compatiblity. For your individual needs, you may chose to keep the version we have currently linked, or update it to the version of the engine build you are using.

**What is important is that you are aware of which version of godot-cpp you have, which you wish to use, and know how to change it.**

You can check the version of your godot-cpp by changing to that directory, typing `git log`, and finding the most recent tag.

```
Terrain3D/godot-cpp$ git log
commit 631cd5fe37d4e6df6e5eb66eb4435feca12708cc (HEAD, tag: godot-4.1.3-stable, 4.1)
...

```

Use one of these steps below to find and select the desired tag or commit hash, then move on to step 4.

You may need to update your godot-cpp before it can find or checkout the latest tags:

```
Terrain3D/godot-cpp$ git fetch
From https://github.com/godotengine/godot-cpp
 * [new tag]         godot-4.0.1-stable -> godot-4.0.1-stable
 * [new tag]         godot-4.0.2-stable -> godot-4.0.2-stable
```

Now we can search for or checkout more recent tags and commits.


### Using tags
On the [Godot-cpp repository page](https://github.com/godotengine/godot-cpp), click the branch selector, then `Tags` to identify available tags that match the Godot engine binary you wish to use. If your engine version is in this list, e.g. `godot-4.0.2-stable`, great, move on to step 4. Otherwise explore the commit history on the website or command line as shown below.

```{image} images/build_tags.png
:target: ../_images/build_tags.png
```


### Using the commit history
If your engine version doesn't have a tag assigned, because it's a custom build, you can look at the [godot-cpp commit history](https://github.com/godotengine/godot-cpp/commits/master) for a commit that syncs the repository to the upstream engine version. Search for entries named `Sync with upstream commit...`.

Eg, from Godot 4.0-stable.

```{image} images/build_commit_history.png
:target: ../_images/build_commit_history.png
```

Clicking the `...` in the middle expands the description which shows that this commit syncs godot-cpp with Godot engine `4.0-stable`. To use this commit, copy the commit string on the right in blue. Click the two overlapping squares on the right to copy the commit hash (`9d1c396`).


### Using the command line
Alternatively, you can use git to search on the command line. Make sure to fetch to update the submodule (step 3). This will search the server (origin) for all commit messages with the string `stable`, allowing you to find the commits. Make sure to grab the commit hash on top (`9d1c396..`), not the upstream commit shown at the bottom, which is from the Godot repository.

```
Terrain3D/godot-cpp$ git log origin -Gstable
commit 9d1c396c54fc3bdfcc7da4f3abcb52b14f6cce8f (HEAD -> master, tag: godot-4.0-stable, origin/master, origin/HEAD, origin/4.0)
Author: R mi Verschelde <rverschelde@gmail.com>
Date:   Wed Mar 1 15:32:44 2023 +0100

    gdextension: Sync with upstream commit 92bee43adba8d2401ef40e2480e53087bcb1eaf1 (4.0-stable)

```


## 4. Check out the correct version
Once you have identified the proper tag or commit string, and you have updated the godot-cpp submodule (step 3), you just need to check it out. If using a commit string, you may use either the full hash or just the first 6-8 characters, so `9d1c396` would also match 4.0-stable.

These examples will change the godot-cpp repository to 4.0-stable and 4.02-stable, respectively:

```
Terrain3D$ cd godot-cpp

Terrain3D/godot-cpp$ git checkout 9d1c396c54fc3bdfcc7da4f3abcb52b14f6cce8f
HEAD is now at 9d1c396 gdextension: Sync with upstream commit 92bee43adba8d2401ef40e2480e53087bcb1eaf1 (4.0-stable)

Terrain3D/godot-cpp$ git checkout godot-4.0.2-stable
Previous HEAD position was 9d1c396 gdextension: Sync with upstream commit 92bee43adba8d2401ef40e2480e53087bcb1eaf1 (4.0-stable)
HEAD is now at 7fb46e9 gdextension: Sync with upstream commit 7a0977ce2c558fe6219f0a14f8bd4d05aea8f019 (4.0.2-stable)

```


## 5. Build the extension

By default `scons` will build the debug library which works for the editor and debug exports. You can add `target=template_release` to build the release version.

```
Terrain3D/godot-cpp$ cd ..

# To build the debug library
Terrain3D$ scons

# To build both debug and release versions sequentially (bash command line)
Terrain3D$ scons && scons target=template_release

```

Upon success you should see something like this at the end:

```
Creating library project\addons\terrain_3d\bin\libterrain.windows.debug.x86_64.lib and object project\addons\terrain_3d\bin\libterrain.windows.debug.x86_64.exp
scons: done building targets.
```


## 6. Set up the extension in Godot

1. Build Terrain3D, then ensure binary libraries exist in `project/addons/terrain_3d/bin`.
2. Close Godot. (Not required the first time, but necessary when updating the files on subsequent builds.)
3. Copy `project/addons/terrain_3d` to your own project folder as `/addons/terrain_3d`. 
4. Run Godot, using the console executable so you can see error messages. Restart when it prompts.
5. In `Project Settings / Plugins`, ensure that Terrain3D is enabled.
6. Select `Project / Reload Current Project` to restart once more.
7. Create or open a 3D scene and add a new Terrain3D node.
8. Select Terrain3D in the Scene panel. In the Inspector, click the down arrow to the right of the `storage` resource and save it as a binary `.res` file. The other resources can be left as is or saved as text `.tres`. These external files can be shared with other scenes.
9. Learn how to properly [set up your textures](texture_prep.md), or skip to [importing data](import_export.md).


## Other Build Options

The `scons` build system has additional useful options. These come from the GDExtension template we are using, so some options may not be supported or work properly for this particular plugin. e.g. The platform.


### Debug Symbols

Build the extension with debug symbols. See [debugging](#debugging-the-source-code) below.
```
scons dev_build=yes
```


### Clean up build files
```
# Linux, other Unix, Git bash on Windows
scons --clean
rm project/addons/terrain_3d/bin/*
find . -iregex '.+\.\(a\|lib\|o\|obj\|os\)' -delete

# Windows
scons --clean
del /q project\addons\terrain_3d\bin\*.*
del /s /q *.a *.lib *.o *.obj *.os
```


### Manually specify the target platform
This plugin supports Windows, Linux and macOS. See [Mobile & Web Support](mobile_web.md) for the status of other platforms.

```
# platform: Target platform (linux|macos|windows|android|ios|javascript)

scons platform=linux
```


### Using C++20
The C++ standard used in Godot and Godot-cpp is C++17. However you may use C++20 for building GDExtensions if desired.

Our [SConstruct](https://github.com/TokisanGames/Terrain3D/blob/main/SConstruct#L52-L56) file has some commented code towards the bottom that will replace the standard.


### See all options
```
# Godot custom build options
scons --help

# Scons application options
scons -H
```


## Troubleshooting

### Debugging the source code
In addition to the [debug logs](troubleshooting.md#debug-logs) dumped to the console, it is possible to debug Godot and Terrain3D, stepping through the full source code for both projects, viewing the callstack, and watching variables. We do it in MSVC regularly with these steps.

* Build Terrain3D with `scons dev_build=yes`.
* Build Godot with `scons debug_symbols=true`.
* Start Godot with the debugger from within the Godot source project, rather than the Terrain3D project.
* The debugger will attach to the Project Manager. After you load your project in the editor, or your game scene, `Debug/Attach to Process...` to reattach the debugger to the new process. Alternatively, adjust your debug startup command so Godot loads your project in the editor or runs your game scene directly and the debugger will attach to it on startup.
* Since you started in the Godot project, upon hitting a breakpoint in Terrain3D, it will ask for the location of the file. Once found, it should have no problem loading the source code.

If you have problems, use `Debug/Windows/Modules` to display the dependent libraries and ensure the symbols for Godot and Terrain3D are loaded.

You can also debug only Terrain3D, using the official Godot binary. You'll be able to view and step through Terrain3D code. Any calls to Godot will be processed, but you won't be able to step through the code or watch variables without the symbols.


### When running scons, I get these errors:

```
Terrain3D$ scons
scons: Reading SConscript files ...

scons: warning: Calling missing SConscript without error is deprecated.
Transition by adding must_exist=False to SConscript calls.
Missing SConscript 'godot-cpp\SConstruct'
File "C:\gd\Terrain3D\SConstruct", line 6, in <module>
AttributeError: 'NoneType' object has no attribute 'Append':
  File "C:\gd\Terrain3D\SConstruct", line 9:
    env.Append(CPPPATH=["src/"])

```

Your godot-cpp directory is probably empty. Review the instructions above for updating the submodule.


### I can build the plugin, however Godot instantly crashes. 
Your godot-cpp version probably does not match your engine version. See section 3 above to learn how to identify and change versions. Test the example project in the next question.


### How can I make sure godot-cpp is the right version and working?
You'll find a test project in `godot-cpp/test/`. Make sure this test project works with your Godot version first, then come back and try Terrain3D again.
  * Build the example plugin by typing `scons` while in the `godot-cpp/test/` directory.
  * Copy `example.gdextension` and `bin` into the root folder of your project.
  * Run Godot. If it crashes, you're on the wrong version, or Godot-cpp has a problem that the maintainers will need to resolve.
  * Create a new scene.
  * Add a new `Example` node. When clicking the node, you should see an `Example` section and `Property From List` and various `Dproperty#` variables in the inspector.
