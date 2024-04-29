extends Node

var add_node = do_add_node() # Hack to have one node on init and not fail at runtime.

var shorthand = $Node
var with_self = self.get_node(^"Node")
var without_self = get_node(^"Node")
var with_cast = get_node(^"Node") as Node
var shorthand_with_cast = $Node as Node

func test():
	print("warn")

func do_add_node():
	var node = Node.new()
	node.name = "Node"
	@warning_ignore("unsafe_call_argument")
	add_child(node)
