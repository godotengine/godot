# https://github.com/godotengine/godot/issues/72948

class Example:
	extends RefCounted

const const_ints : Array[int] = [1, 2, 3]

func test():
	var ints: Array[int] = [1, 2, 3]
	var strings: Array[String] = ["4", "5", "6"]

	var ints_concatenated: Array[int] = ints + ints
	var strings_concatenated: Array[String] = strings + strings
	var untyped_concatenated: Array = ints + strings
	var const_ints_concatenated: Array[int] = const_ints + const_ints

	print(ints_concatenated.get_typed_builtin())
	print(strings_concatenated.get_typed_builtin())
	print(untyped_concatenated.get_typed_builtin())
	print(const_ints_concatenated.get_typed_builtin())

	var objects: Array[Object] = []
	var objects_concatenated: Array[Object] = objects + objects
	print(objects_concatenated.get_typed_class_name())

	var examples: Array[Example] = []
	var examples_concatenated: Array[Example] = examples + examples
	print(examples_concatenated.get_typed_script() == Example)
