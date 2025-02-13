extends Node

@onready var shorthand = $Node
@onready var call_no_cast = get_node(^"Node")
@onready var shorthand_with_cast = $Node as Node
@onready var call_with_cast = get_node(^"Node") as Node

func _init():
	var node := Node.new()
	node.name = "Node"
	add_child(node)

func test():
	# Those are expected to be `null` since `_ready()` is never called on tests.
	prints("shorthand", shorthand)
	prints("call_no_cast", call_no_cast)
	prints("shorthand_with_cast", shorthand_with_cast)
	prints("call_with_cast", call_with_cast)
