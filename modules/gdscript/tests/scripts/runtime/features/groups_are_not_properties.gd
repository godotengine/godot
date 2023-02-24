# https://github.com/godotengine/godot/issues/73843
extends RefCounted

@export_group("Resource")
@export_category("RefCounted")

func test():
	var res = Resource.new()
	var ref = RefCounted.new()
	prints("Resource class not shadowed:", res is Resource)
	prints("RefCounted class not shadowed:", ref is RefCounted)
