extends Node

@oninstantiated var shorthand = $Node
@oninstantiated var call_no_cast = get_node(^"Node")
@oninstantiated var shorthand_with_cast = $Node as Node
@oninstantiated var call_with_cast = get_node(^"Node") as Node

func _init():
	var node := Node.new()
	node.name = "Node"
	add_child(node)

func test():
	# Those are expected to be `null` since `_scene_instantiated()` is never called on tests.
	prints("shorthand", shorthand)
	prints("call_no_cast", call_no_cast)
	prints("shorthand_with_cast", shorthand_with_cast)
	prints("call_with_cast", call_with_cast)
