# https://github.com/godotengine/godot/issues/72501
extends Node

func test():
	prints("before", process_mode)
	process_mode = PROCESS_MODE_PAUSABLE
	prints("after", process_mode)

	var node := Node.new()
	add_child(node)
	prints("before", node.process_mode)
	node.process_mode = PROCESS_MODE_PAUSABLE
	prints("after", node.process_mode)
