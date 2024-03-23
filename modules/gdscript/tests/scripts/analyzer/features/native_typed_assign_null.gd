extends Node

func test():
	var typed: Variant = get_node_or_null("does_not_exist")
	var untyped = null
	var node_1: Node = typed
	var node_2: Node = untyped
	var node_3 = typed
	var node_4 = untyped
	print(typed)
	print(untyped)
	print(node_1)
	print(node_2)
	print(node_3)
	print(node_4)
