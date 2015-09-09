#This script is for testing EMScripten templates. Only tested on Mac OSX 10.10.
#Based on https://github.com/okamstudio/godot/wiki/compiling_batch_templates

#Need to set path to EMScripten
export EMSCRIPTEN_ROOT=~/Dev/emsdk_portable/emscripten/1.34.1

#Build OSX first so shaders are included.

~/bin/scons -j 4 p=osx

# EMScripten
# Note: Changed a couple of 'cp' to 'sed' in order to change file references to 
#       what platform/javascript/export/export.cpp expects.
#       Also note that more recent Emscripten emits a *.html.mem file.

~/bin/scons -j 4 p=javascript target=release
sed -e 's/godot.*js/godot\.js/' bin/godot.javascript.opt.html > godot.html
sed -e 's/godot.javascript.opt.html.mem/godot.html.mem/g' bin/godot.javascript.opt.js > godot.js
cp bin/godot.javascript.opt.html.mem godot.html.mem
cp tools/html_fs/filesystem.js .
zip javascript_release.zip godot.html godot.js godot.html.mem filesystem.js
cp javascript_release.zip ~/.godot/templates

~/bin/scons -j 4 p=javascript target=debug
#cp bin/godot.javascript.debug.html godot.html
sed -e 's/godot.*js/godot\.js/' bin/godot.javascript.debug.html > godot.html
#cp bin/godot.javascript.debug.js godot.js
sed -e 's/godot.javascript.debug.html.mem/godot.html.mem/g' bin/godot.javascript.debug.js > godot.js
cp bin/godot.javascript.debug.html.mem godot.html.mem
cp tools/html_fs/filesystem.js .
zip javascript_debug.zip godot.html godot.js godot.html.mem filesystem.js
cp javascript_debug.zip ~/.godot/templates

rm godot.html
rm godot.js
rm godot.html.mem
rm filesystem.js

# Just for quick testing, avoid having to manually re-export from editor every time, 
# just one initial export as 'godot.html' is required.
#cp bin/godot.javascript.debug.js ~/Downloads/godot_demos-1.1stable/builds/platformer3d/godot.js
#cp bin/godot.javascript.debug.html.mem ~/Downloads/godot_demos-1.1stable/builds/platformer3d

rm javascript_release.zip
rm javascript_debug.zip


