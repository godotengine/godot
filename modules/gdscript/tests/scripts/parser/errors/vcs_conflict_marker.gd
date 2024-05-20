# The VCS conflict marker has only 6 `=` signs instead of 7 to prevent editors like
# Visual Studio Code from recognizing it as an actual VCS conflict marker.
# Nonetheless, the GDScript parser is still expected to find and report the VCS
# conflict marker error correctly.

<<<<<<< HEAD
Hello world
======
Goodbye
>>>>>>> 77976da35a11db4580b80ae27e8d65caf5208086

func test():
	pass
