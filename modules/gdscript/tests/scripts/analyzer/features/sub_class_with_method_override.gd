# https://github.com/godotengine/godot/pull/57510
extends "base_class.gd"

func virtual_method(_ext: external_script) -> String:
	return "Subclass"

func test():
	var external_instance = external_script.new()
	print("virtual_method call: " + virtual_method(external_instance))
