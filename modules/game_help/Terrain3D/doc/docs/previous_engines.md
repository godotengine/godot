Using Previous Engine Versions
==============================

In general, you want to match the compiled version of your godot-cpp folder to the engine version. As of Godot 4.1, using an older godot-cpp version should work in future engine versions within the same minor number. (e.g. 4.1.x). And depending on API changes, it may work with the next minor version as well (e.g. 4.2.x). A compiled plugin version won't work with previous versions of the engine.

If you want to build the current version of Terrain3D for a previous version of the engine, it's generally possible just by changing your godot-cpp commit and rebuilding, as explained in [Building from Source](building_from_source.md). However, occasionally there are API changes required to get the plugin working in future engine versions. We are attempting to document those here so they can be undone or worked around by you to support older an engine version.

The last tagged release is the version of Terrain3D used when that version of the engine was in use, primarily for reference or testing.

**Example:** Say you wish to use the current version of Terrain3D with Godot 4.0.6. You could build the 4.0.3 tag with a 4.0.6 godot-cpp, and make sure that works with 4.0.6. Then you'd checkout the `main` branch, and undo the commits listed for 4.2, 4.1, etc. Some of those changes might be very easily removed, such as the first from 4.0 to 4.1. Others will utilize new features of Godot that have been introduced in later versions. In those cases, you'll need to wait until Godot backports those features to your version (primary use case), or modify Terrain3D to get it working (likely very difficult).

|Version|Commit to undo|Last tagged release|
|-----|----|----|
|4.0 -> 4.1|[da4551](https://github.com/TokisanGames/Terrain3D/commit/da455147d18674d02ba4b88bd575b58de472c617)|[4.0.3](https://github.com/TokisanGames/Terrain3D/releases/tag/v0.8-alpha_gd4.0.3)|
