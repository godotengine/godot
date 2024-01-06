## Lua Libraries
This is a code gen module which allows you to statically compile a lua C library into this addon. Its as simple as downloading the source for the addon, and adding it as a folder with the library name in this folder.

It is important that the folder name be exactly the library name in lowercase. No version numbers included.

The code gen will auto detect it next time you build the addon. Either as a module or for GDExtension.

This is very new and potentially very prone to failure. So for lpeg has been tested and confirmed to work.