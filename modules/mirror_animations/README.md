# Godot - Mirror Animations

This module adds a "Mirror" option to animations imported via FBX / GLTF etc.  
It only works for humanoid rigs and only if the armatures rest pose is in a proper T-Pose.

It's not tested with other poses, however, not having a proper rest pose will break the animation.  
This module was written for Godot v4.2+ and might not compile for version lower than that.  

## Setup

1. Clone the godot engine somewhere to your local machine
2. Follow the `Building from source` guidelines [here](https://docs.godotengine.org/en/stable/contributing/development/compiling/index.html)
3. Add this repo as a submodule to your project:  
   `cd /path/to/godot/`
   `git submodule add -b main https://github.com/Lunatix89/godot-mirror-animations modules/mirror_animations`
4. Rebuild the engine

## Guidelines for typical unity animations 

Unity animations are typically split up in a character in T-Pose and several, small animation files which won't contain a skin, just the armature with the specific animation.  
There are also animations out there which don't have a proper T-Pose set for the individual animation, the rest pose is just the first animation frame.  

To work around this:
1. Buy and install [Better FBX Importer & Exporter](https://blendermarket.com/products/better-fbx-importer--exporter) from blender market.  
   This plugin will solve wrong bone orientations which can occur with the default FBX import plugin as well as having a special import option which we later need.
2. Import the actual character which should has it's rest pose set to T-Pose.
   Make sure that it has a proper armature and rest pose set up.
3. Select the armature
4. Import the desired animation with `Animations Options / Attach to Selected Armature` checked.
   If the animation files also import an armature and model, select and delete them
5. Export as `glTF 2.0` - this is important, do not export to FBX again
6. Import into Godot
7. Open the advanced import options dialog
8. Select the desired animation
9. Check `Mirror` on the animation
10. Click `Reimport`

Please note that I can't guarantee that this will work for every animation file as I only have a limited amount of files to test with.  

