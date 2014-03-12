#*************************************************************************/
#*                       This file is part of:                           */
#*                           GODOT ENGINE                                */
#*                    http://www.godotengine.org                         */
#*************************************************************************/
# Simple makefile to give support for external C/C++ IDEs                */
#*************************************************************************/

# Default build
all: debug

# Release Build
release:
	scons target="release" bin/godot

# Profile Build
profile:
	scons target="profile" bin/godot

# Debug Build
debug:
	# Debug information (code size gets severely affected):
	# g: Default (same as g2)
	# g0: no debug info
	# g1: minimal info
	# g3: maximal info
	scons target="debug" CCFLAGS="-g" bin/godot

clean:
	scons -c bin/godot
