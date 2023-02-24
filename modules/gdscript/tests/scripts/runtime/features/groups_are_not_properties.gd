# https://github.com/godotengine/godot/issues/73843
extends RefCounted

@export_group("Resource")
@export_category("RefCounted")

func test():
	prints("Not shadowed", Resource.new())
	prints("Not shadowed", RefCounted.new())
