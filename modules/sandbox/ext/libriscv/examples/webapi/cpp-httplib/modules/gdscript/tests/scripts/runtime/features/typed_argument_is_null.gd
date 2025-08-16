# https://github.com/godotengine/godot/issues/72967

class CustomNode:
	extends Node

	static func test_custom_node(n: CustomNode):
		if not n:
			print("null node")

func test():
	test_typed_argument_is_null()

func get_custom_node() -> CustomNode:
	return null

func test_typed_argument_is_null():
	var node: Node = Node.new()
	print_node_name(node.get_parent())
	node.free()
	test_custom_node()

func test_custom_node():
	CustomNode.test_custom_node(get_custom_node())

func print_node_name(n: Node):
	if not n:
		print("null node")
